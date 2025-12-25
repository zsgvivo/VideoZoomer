# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
import re
import numpy as np
import wandb
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from collections import defaultdict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.trainer.ppo.monkey_patch import create_colocated_worker_cls_patch
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torch.utils.data import RandomSampler, SequentialSampler
from verl.utils.dataset.multimodal_dataset import MultiModalDataset
from torchdata.stateful_dataloader import StatefulDataLoader
import json
import time
from verl.workers.rollout.vllm_rollout.async_server import AsyncLLMEngineManager


WorkerType = Type[Worker]

def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(
        tensors=item.batch,  # TensorDict is already in correct format
        non_tensors=item.non_tensor_batch,  # Dict is already in correct format 
        meta_info=item.meta_info
    )

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics

def find_ones_segments_lengths(tensor):
    # 获取所有值为1的索引
    indices = torch.nonzero(tensor == 1, as_tuple=True)[0]
    if len(indices) == 0:
        return []
    # 计算相邻索引差值，找到分割点（差值>1的位置）
    diffs = indices[1:] - indices[:-1]
    split_positions = torch.where(diffs > 1)[0] + 1  # +1转换为右边界
    # 分割成连续段并计算每段长度
    segments = torch.tensor_split(indices, split_positions.tolist())
    lengths = [segment.size(0) for segment in segments]
    return lengths

def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]
    multi_turn_response_mask = batch.batch['multi_turn_response_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    text_response_length = multi_turn_response_mask.sum(-1).float()
    text_response_length_lst = [find_ones_segments_lengths(row) for row in multi_turn_response_mask]
    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
        text_response_length=text_response_length,
        text_response_length_lst=text_response_length_lst,
    )


def compute_data_metrics(batch, use_critic=True, **kwargs):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    if 'format_scores' in batch.batch:
        format_reward = batch.batch['format_scores']
        accuracy_reward = batch.batch['acc_scores']
    else:
        format_reward = torch.tensor(0.0)
        accuracy_reward = torch.tensor(0.0)

    success_tool_call_mask = batch.batch["success_tool_call"]
    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]    # 65536

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']
    text_response_length = response_info['text_response_length']
    text_response_length_lst = response_info['text_response_length_lst']
    text_response_length_clip_flag = [any([(l == kwargs['max_response_length']) for l in item]) for item in text_response_length_lst]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean/format':
            torch.mean(format_reward).detach().item(),
        'critic/rewards/mean/accuracy':
            torch.mean(accuracy_reward).detach().item(),
        'critic/rewards/mean/tool_call_format':
            torch.mean(format_reward[success_tool_call_mask]).detach().item(),
        'critic/rewards/mean/tool_call_accuracy':
            torch.mean(accuracy_reward[success_tool_call_mask]).detach().item(),
        'critic/rewards/mean/no_tool_call_format':
            torch.mean(format_reward[~success_tool_call_mask]).detach().item(),
        'critic/rewards/mean/no_tool_call_accuracy':
            torch.mean(accuracy_reward[~success_tool_call_mask]).detach().item(),
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/var':
            torch.var(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        'response_length/text_mean':
            torch.mean(text_response_length).detach().item(),
        'response_length/text_max':
            torch.max(text_response_length).detach().item(),
        'response_length/text_min':
            torch.min(text_response_length).detach().item(),
        'response_length/text_clip_ratio':
            sum(text_response_length_clip_flag) / len(text_response_length_lst),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

        # TODO: support each role have individual ray_worker_group_cls,
        # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.is_qwen = self.config.actor_rollout_ref.model.is_qwen
        if self.is_qwen != ('qwen' in self.config.actor_rollout_ref.model.path.lower()):
            print('WARNING: is_qwen is not consistent with model path')
        self.tool_call = self.config.actor_rollout_ref.rollout.tool_call

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()
    def _dump_generations(self, inputs, outputs, scores, dump_path, reward_extra_infos_dict=None):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }
        if reward_extra_infos_dict:
            for k, v in reward_extra_infos_dict.items():
                if len(v) == n:
                    base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: str(v[i]) for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        if isinstance(self.config.data.train_files, str):
            assert self.config.data.train_files.endswith('.yaml') or self.config.data.train_files.endswith('.json')
            self.train_dataset = MultiModalDataset(
                data_path=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.prompt_key,
                image_key=self.config.data.get('image_key', 'images'),
                max_prompt_length=self.config.data.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=self.config.data.get('return_raw_chat', False),
                truncation='error',
                system_prompt=self.config.data.get('system_prompt', None),
                post_prompt=self.config.data.get('post_prompt', None),
                is_qwen=self.is_qwen,
                max_pixels=self.config.data.get('max_pixels', 2048 * 2048),
                min_pixels = self.config.data.get('min_pixels', 512 * 512),
                pass_video_as_frames=self.config.data.get('pass_video_as_frames', True),
                video_fps=self.config.data.get('video_fps', 1),
                frames_upbound=self.config.data.get('frames_upbound', 30),
                fast_seek=self.config.data.get('fast_seek', False),
                random_fps=self.config.data.get('random_fps', None),
            )
        else:
            self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                            tokenizer=self.tokenizer,
                                            processor=self.processor,
                                            prompt_key=self.config.data.prompt_key,
                                            image_key=self.config.data.get('image_key', 'images'),
                                            max_prompt_length=self.config.data.max_prompt_length,
                                            filter_prompts=True,
                                            return_raw_chat=self.config.data.get('return_raw_chat', False),
                                            truncation='error',
                                            system_prompt=self.config.data.get('system_prompt', None),
                                            post_prompt=self.config.data.get('post_prompt', None),
                                            is_qwen=self.is_qwen)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        train_batch_size = self.config.data.train_batch_size
        if self.config.trainer.rejection_sample:
            train_batch_size *= self.config.trainer.rejection_sample_multiplier
            train_batch_size = int(train_batch_size)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=train_batch_size,
                                                   num_workers=16,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler,
                                                   prefetch_factor=1,
                                                   persistent_workers=True,)

        if self.config.data.val_files != 'None':
            if isinstance(self.config.data.val_files, str):
                assert self.config.data.val_files.endswith('.yaml') or self.config.data.val_files.endswith('.json')
                self.val_dataset = MultiModalDataset(
                    data_path=self.config.data.val_files,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    prompt_key=self.config.data.prompt_key,
                    image_key=self.config.data.get('image_key', 'images'),
                    max_prompt_length=self.config.data.max_prompt_length,
                    filter_prompts=True,
                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                    truncation='error',
                    system_prompt=self.config.data.get('system_prompt', None),
                    post_prompt=self.config.data.get('post_prompt', None),
                    is_qwen=self.is_qwen,
                    max_pixels=self.config.data.get('max_pixels', 2048 * 2048),
                    min_pixels = self.config.data.get('min_pixels', 512 * 512),
                    pass_video_as_frames=self.config.data.get('pass_video_as_frames', True),
                    video_fps=self.config.data.get('video_fps', 1),
                    frames_upbound=self.config.data.get('frames_upbound', 30),
                    fast_seek=self.config.data.get('fast_seek', False),
                )
            else:
                self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                            tokenizer=self.tokenizer,
                                            processor=self.processor,
                                            prompt_key=self.config.data.prompt_key,
                                            image_key=self.config.data.get('image_key', 'images'),
                                            max_prompt_length=self.config.data.max_prompt_length,
                                            filter_prompts=True,
                                            return_raw_chat=self.config.data.get('return_raw_chat', False),
                                            truncation='error',
                                            system_prompt=self.config.data.get('system_prompt', None),
                                            post_prompt=self.config.data.get('post_prompt', None),
                                            is_qwen=self.is_qwen)
            val_batch_size = self.config.data.val_batch_size or len(self.val_dataset)
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                # Validation datasets are sent to inference engines as a whole batch,
                # which will schedule the memory themselves.
                batch_size=val_batch_size,
                num_workers=16,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
                persistent_workers=True,
                )

        assert len(self.train_dataloader) >= 1
        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_train_generations_to_wandb(self, batch, reward_tensor):
        """Log a table of train samples to wandb"""
        generations_to_log = self.config.trainer.train_generations_to_log_to_wandb
        
        if generations_to_log == 0:
            return
        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return
        input_ids = batch.batch['prompts'][:generations_to_log]
        response_ids = batch.batch['responses'][:generations_to_log]
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = batch.non_tensor_batch['ground_truth'].tolist()[:generations_to_log]
        scores = reward_tensor.max(dim=-1)[0][:generations_to_log]
        tool_match = r'<video_zoom>.*</video_zoom>'
        tool_use = [1 if re.search(tool_match, x) else 0 for x in outputs]
        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, tool_use, ground_truth, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text
        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"use_tool_{i+1}", f"ground_truth_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])
        if not hasattr(self, 'train_table'):
            # Initialize the table on first call
            self.train_table = wandb.Table(columns=columns)
        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.train_table.data)
        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)
        if generations_to_log > len(samples):
            for _ in range(generations_to_log - len(samples)):
                row_data.extend(['', '', '', torch.tensor(0)])
        new_table.add_data(*row_data)
        # Update reference and log
        wandb.log({"train/generations": new_table}, step=self.global_steps)
        self.train_table = new_table

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, ground_truth, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        tool_match = r'<video_zoom>.*</video_zoom>'
        tool_use = [1 if re.search(tool_match, x) else 0 for x in outputs]
        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs,tool_use, ground_truth, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"use_tool_{i+1}",f"ground_truth_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])
        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _maybe_log_tool_call_metrics(self, batch, metrics):
        response_ids = batch.batch['responses']
        accuracy_scores = batch.batch['acc_scores']
        multi_modal_data = batch.non_tensor_batch.get('multi_modal_data', None)
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        apply_tool_call_list = []
        success_tool_call_list = batch.batch['success_tool_call'].tolist()
        tool_call_frame_num = batch.batch["tool_call_frame_num"].tolist()
        tool_call_generations_to_log_to_wandb = self.config.trainer.tool_call_generations_to_log_to_wandb
        successful_tool_call_to_log = []
        for idx, success_tool_call in enumerate(success_tool_call_list):
            if self.config.actor_rollout_ref.rollout.tool_call == 'video_zoom':
                tool_call_pattern = re.compile(r'<video_zoom>(.*?)</video_zoom>', re.DOTALL)
                apply_tool_call = len(tool_call_pattern.findall(responses[idx])) > 0
            else:
                raise NotImplementedError
            if success_tool_call and len(successful_tool_call_to_log) < tool_call_generations_to_log_to_wandb:
                successful_tool_call_to_log.append(idx)
            apply_tool_call_list.append(apply_tool_call)
        if batch.meta_info.get('validate', False):  # val
            metrics['val/tool_call/success_tool_call_ratio_per_batch'] = sum(success_tool_call_list) / len(success_tool_call_list)
            metrics['val/tool_call/success_tool_call_rate'] = sum(success_tool_call_list) / (sum(apply_tool_call_list) + 1e-4)
            metrics['val/tool_call/success_tool_call_correct_answer_ratio'] = (accuracy_scores[success_tool_call_list] == 1).sum().item() / (sum(success_tool_call_list) + 1e-4)
            metrics['val/tool_call/tool_call_frame_num'] = sum(tool_call_frame_num) / (len(tool_call_frame_num) + 1e-4)
            round_nums_counter = defaultdict(int)
            round_correct_counter = defaultdict(int)
            batch_rounds_num_list = []
            for idx in range(len(batch.batch['multi_turn_response_mask'])):
                prompt_and_response_ids = torch.cat([batch.batch['prompts'][idx], batch.batch['responses'][idx]], dim=0)
                response_ids = prompt_and_response_ids[batch.batch['multi_turn_response_mask'][idx] == 1]
                response = self.tokenizer.decode(response_ids)
                rounds_num = response.count("<|im_end|>")
                batch_rounds_num_list.append(rounds_num)
                round_nums_counter[rounds_num] += 1
                assert rounds_num >= 1, f"response: {response}"
                assert rounds_num <= self.config.actor_rollout_ref.rollout.max_generation_round, f"rounds_num {rounds_num} should not be greater than max_generation_round {self.config.actor_rollout_ref.rollout.max_generation_round}, response: {response}"
            batch.batch['rounds_num'] = torch.tensor(batch_rounds_num_list, dtype=torch.int32)

            for idx in range(len(batch.batch['acc_scores'])):
                rounds_num = batch.batch['rounds_num'][idx].item()
                round_correct_counter[rounds_num] += (batch.batch['acc_scores'][idx].item() > 0)
            for idx in range(self.config.actor_rollout_ref.rollout.max_generation_round):
                metrics[f'val/round_distribution/round_{idx+1}_ratio'] = round_nums_counter[idx+1] / sum(round_nums_counter.values())
                metrics[f'val/round_distribution/round_{idx+1}_acc_ratio'] = round_correct_counter[idx+1] / (round_nums_counter[idx+1] + 1e-5)
            # log tool call trajectory to wandb
            if len(successful_tool_call_to_log) > 0 and multi_modal_data is not None:
                input_ids = batch.batch['prompts'][successful_tool_call_to_log]
                response_ids = batch.batch['responses'][successful_tool_call_to_log]
                inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                ground_truth = batch.non_tensor_batch['ground_truth'][successful_tool_call_to_log].tolist()
                scores = accuracy_scores[successful_tool_call_to_log]
                multi_modal_datas = multi_modal_data[successful_tool_call_to_log]
                samples = list(zip(inputs, outputs, ground_truth, scores, multi_modal_datas))
                samples.sort(key=lambda x: x[0])  # Sort by input text
                columns = ["step"] + [f"input", f"output", f"ground_truth", f"score"] + [f"image_{i}" for i in range(self.config.actor_rollout_ref.rollout.limit_mm_per_prompt['image'])]
                if not hasattr(self, 'val_tool_table'):
                    # Initialize the table on first call
                    self.val_tool_table = wandb.Table(columns=columns)
                # Create a new table with same columns and existing data
                # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                new_table = wandb.Table(columns=columns, data=self.val_tool_table.data)
                # Add new row with all data
                for sample in samples:
                    row_data = []
                    row_data.append(self.global_steps)
                    (inputs, outputs, ground_truth, score, multi_modal_data) = sample
                    row_data.extend([inputs, outputs, ground_truth, score])
                    for image in multi_modal_data['image']:
                        row_data.append(wandb.Image(image))
                    while len(row_data) < len(columns):
                        row_data.append(wandb.Image(np.zeros((10, 10, 3))))
                    row_data = row_data[:len(columns)]
                    new_table.add_data(*row_data)
                # Update reference and log
                wandb.log({"val/tool_call": new_table}, step=self.global_steps)
                self.val_tool_table = new_table
        else:   # train
            num_rollouts = self.config.actor_rollout_ref.rollout.n
            sample_level_success_tool_call_list = [True if any(success_tool_call_list[i : i + num_rollouts]) else False for i in range(0, len(success_tool_call_list), num_rollouts)]   # 对每个样本,只要Rollout里面有一个调用了工具,那么就算这个样本调了工具
            metrics['tool_call/success_tool_call_ratio_per_batch'] = sum(success_tool_call_list) / len(success_tool_call_list)
            metrics['tool_call/success_tool_call_ratio_per_sample'] = sum(sample_level_success_tool_call_list) / len(sample_level_success_tool_call_list)
            metrics['tool_call/success_tool_call_rate'] = sum(success_tool_call_list) / (sum(apply_tool_call_list) + 1e-4)
            metrics['tool_call/success_tool_call_correct_answer_ratio'] = (accuracy_scores[success_tool_call_list] == self.config.reward_model.acc_reward_weight).sum().item() / (sum(success_tool_call_list) + 1e-4)
            metrics['tool_call/tool_call_frame_num'] = sum(tool_call_frame_num) / (len(tool_call_frame_num) + 1e-4)
            # log different scores combination ratio
            group_accuracy_reward_tensor = accuracy_scores.view(-1, num_rollouts)
            num_samples = group_accuracy_reward_tensor.shape[0]
            group_success_tool_call_mask = torch.tensor(success_tool_call_list).view(-1, num_rollouts)
            use_tool_correct_count = ((group_accuracy_reward_tensor == self.config.reward_model.acc_reward_weight) & group_success_tool_call_mask).sum(dim=1)  # (bs,)
            direct_answer_correct_count = ((group_accuracy_reward_tensor == self.config.reward_model.acc_reward_weight) & (~group_success_tool_call_mask)).sum(dim=1)  # (bs,)
            wrong_answer_count = (group_accuracy_reward_tensor == 0).sum(dim=1) # (bs,)
            # Obtain mask
            use_tool_correct_mask = use_tool_correct_count > 0
            direct_answer_correct_mask = direct_answer_correct_count > 0
            wrong_answer_mask = wrong_answer_count > 0
            # Due to rejection sample, here we will not encounter all_correct | all_wrong cases
            wrong_answer_use_tool_correct_num = (use_tool_correct_mask & wrong_answer_mask & (~direct_answer_correct_mask)).sum().item()  # [0, 1 - penalty]
            wrong_answer_direct_answer_correct_num = (direct_answer_correct_mask & wrong_answer_mask & (~use_tool_correct_mask)).sum().item()  # [0, 1]
            use_tool_correct_direct_answer_correct_num = (use_tool_correct_mask & direct_answer_correct_mask & (~wrong_answer_mask)).sum().item()  # [1 - penalty, 1]
            wrong_answer_use_tool_correct_direct_answer_correct_num = (use_tool_correct_mask & direct_answer_correct_mask & wrong_answer_mask).sum().item() # [0, 1 - penalty, 1]
            metrics['tool_call/wrong_answer_use_tool_correct_ratio'] = wrong_answer_use_tool_correct_num / num_samples
            metrics['tool_call/wrong_answer_direct_answer_correct_ratio'] = wrong_answer_direct_answer_correct_num / num_samples
            metrics['tool_call/use_tool_correct_direct_answer_correct_ratio'] = use_tool_correct_direct_answer_correct_num / num_samples
            metrics['tool_call/wrong_answer_use_tool_correct_direct_answer_correct_ratio'] = wrong_answer_use_tool_correct_direct_answer_correct_num / num_samples
            # log tool call trajectory to wandb
            if successful_tool_call_to_log and multi_modal_data is not None:
                total_scores = batch.batch['token_level_scores'].sum(-1)
                input_ids = batch.batch['prompts'][successful_tool_call_to_log]
                response_ids = batch.batch['responses'][successful_tool_call_to_log]
                inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                ground_truth = batch.non_tensor_batch['ground_truth'][successful_tool_call_to_log].tolist()
                scores = total_scores[successful_tool_call_to_log]
                multi_modal_datas = multi_modal_data[successful_tool_call_to_log]
                samples = list(zip(inputs, outputs, ground_truth, scores, multi_modal_datas))
                samples.sort(key=lambda x: x[0])  # Sort by input text
                columns = ["step"] + [f"input", f"output", f"ground_truth", f"score"] + [f"image_{i}" for i in range(self.config.actor_rollout_ref.rollout.limit_mm_per_prompt['image'])]
                if not hasattr(self, 'tool_table'):
                    # Initialize the table on first call
                    self.tool_table = wandb.Table(columns=columns)
                # Create a new table with same columns and existing data
                # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                new_table = wandb.Table(columns=columns, data=self.tool_table.data)
                # Add new row with all data
                for sample in samples:
                    row_data = []
                    row_data.append(self.global_steps)
                    (inputs, outputs, ground_truth, score, multi_modal_data) = sample
                    row_data.extend([inputs, outputs, ground_truth, score])
                    for image in multi_modal_data['image']:
                        row_data.append(wandb.Image(image))
                    while len(row_data) < len(columns):
                        row_data.append(wandb.Image(np.zeros((10, 10, 3))))
                    row_data = row_data[:len(columns)]
                    new_table.add_data(*row_data)
                # Update reference and log
                wandb.log({"train/tool_call": new_table}, step=self.global_steps)
                self.tool_table = new_table
        return metrics
    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        if self.config.reward_model.log_rewards_separately:
            format_reward_tensor_lst = []
            acc_reward_tensor_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_ground_truth = []
        sample_scores = []
        image_num_list = []
        success_tool_call = []
        tool_call_frame_num = []
        for idx, test_data in enumerate(self.val_dataloader):
            print(f"Validating on batch {idx} / {len(self.val_dataloader)}")
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                                                    dtype=object)
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_ground_truth.extend(test_batch.non_tensor_batch['ground_truth']) # FIXME: Currently do not support ['reward_model']['ground_truth']

            if 'multi_modal_data' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'video_files'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            if self.config.actor_rollout_ref.rollout.val_n > 1:
                test_gen_batch.meta_info['do_sample'] = True

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size * self.config.actor_rollout_ref.rollout.val_n)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_n, interleave=True)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            from verl.workers.reward_manager import CustomRewardManager, NaiveMultiThreadsV2RewardManager, NaiveMultiThreadsRewardManager
            if isinstance(self.val_reward_fn, CustomRewardManager):
                reward_tensor, acc_reward_tensor, format_reward_tensor = self.val_reward_fn(test_batch)
            elif isinstance(self.val_reward_fn, NaiveMultiThreadsV2RewardManager) or isinstance(self.val_reward_fn, NaiveMultiThreadsRewardManager):
                reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids = self.val_reward_fn(test_batch)
            else:
                reward_tensor = self.val_reward_fn(test_batch)
            if self.config.reward_model.log_rewards_separately:
                if isinstance(self.val_reward_fn, CustomRewardManager) or isinstance(self.val_reward_fn, NaiveMultiThreadsV2RewardManager) or isinstance(self.val_reward_fn, NaiveMultiThreadsRewardManager):
                    format_reward_tensor_lst.append(format_reward_tensor)
                    acc_reward_tensor_lst.append(acc_reward_tensor)
                else:
                    format_reward_tensor = self.val_reward_fn.cal_format_reward_for_logging(test_batch)
                    acc_reward_tensor = self.val_reward_fn.cal_acc_reward_for_logging(test_batch)
                    format_reward_tensor_lst.append(format_reward_tensor)
                    acc_reward_tensor_lst.append(acc_reward_tensor)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            image_num_list.extend([len(item['image']) for item in test_batch.non_tensor_batch['multi_modal_data']])
            success_tool_call.extend([len(item['image']) > prompt_img_num for item, prompt_img_num in zip(test_batch.non_tensor_batch['multi_modal_data'], test_batch.non_tensor_batch['prompt_img_num'])])
            tool_call_frame_num.extend([len(item['image']) - prompt_img_num for item, prompt_img_num in zip(test_batch.non_tensor_batch['multi_modal_data'], test_batch.non_tensor_batch['prompt_img_num'])])
            if self.config.trainer.val_only and self.config.trainer.save_mem:
                test_batch.non_tensor_batch.pop('multi_modal_data')
                assert 'multi_modal_data' not in test_batch.non_tensor_batch.keys()
            if idx == 0:
                total_test_batch = test_batch
            else:
                total_test_batch = DataProto.concat([total_test_batch, test_batch])

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, ground_truth=sample_ground_truth, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        if self.config.reward_model.log_rewards_separately:
            format_reward_tensor = torch.cat(format_reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            acc_reward_tensor = torch.cat(acc_reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)

        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}/total'] = np.mean(rewards)

        if self.config.reward_model.log_rewards_separately:
            data_source_format_reward = {}
            data_source_acc_reward = {}
            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_format_reward:
                    data_source_format_reward[data_source] = []
                data_source_format_reward[data_source].append(format_reward_tensor[i].item())
                if data_source not in data_source_acc_reward:
                    data_source_acc_reward[data_source] = []
                data_source_acc_reward[data_source].append(acc_reward_tensor[i].item())
            for data_source, rewards in data_source_format_reward.items():
                metric_dict[f'val/test_score/{data_source}/format_reward'] = np.mean(rewards)
            for data_source, rewards in data_source_acc_reward.items():
                metric_dict[f'val/test_score/{data_source}/acc_reward'] = np.mean(rewards)
        total_test_batch.meta_info['validate'] = True
        total_test_batch.batch['acc_scores'] = acc_reward_tensor
        total_test_batch.batch['format_scores'] = format_reward_tensor
        if self.tool_call in ['video_zoom']:
            total_test_batch.batch["success_tool_call"] = torch.tensor(success_tool_call)
            total_test_batch.batch["tool_call_frame_num"] = torch.tensor(tool_call_frame_num)
            metric_dict = self._maybe_log_tool_call_metrics(batch=total_test_batch, metrics=metric_dict)
        response_length = total_test_batch.batch['responses'].shape[-1]

        prompt_mask = total_test_batch.batch['attention_mask'][:, :-response_length]
        response_mask = total_test_batch.batch['attention_mask'][:, -response_length:]
        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                dump_path=val_data_dir,
                reward_extra_infos_dict={
                    "tool_call": total_test_batch.batch["success_tool_call"].cpu().tolist(),
                    "acc_score": total_test_batch.batch["acc_scores"].cpu().tolist(),
                    "format_score": total_test_batch.batch["format_scores"].cpu().tolist(),
                    "prompt_len": prompt_mask.sum(-1).float().cpu().tolist(),
                    "response_len": response_mask.sum(-1).float().cpu().tolist(),
                    "data_source": data_sources.tolist(),
                    "frame_num": image_num_list,
                    "extra_info": total_test_batch.non_tensor_batch['extra_info'].tolist()
                }
            )

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls_patch(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMEngineManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
        print("init async worker")

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True) and self.config.trainer.test_freq != -1:
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        solve_none = 0
        solve_all = 0
        solve_acc_none = 0
        solve_acc_all = 0
        solve_format_none = 0
        solve_format_all = 0
        success_tool_call_per_batch = 0
        success_tool_call_per_sample = 0
        solve_acc = 0
        solve_format = 0
        totals = 0
        invalid_num = 0
        effective_num = 0
        round_nums_counter = defaultdict(int)
        round_correct_counter = defaultdict(int)
        last_step_end_time = time.time()
        if self.tool_call in ['video_zoom']:
            use_tool_and_direct_answer_num = 0
            wrong_and_use_tool_num = 0
            wrong_and_direct_answer_num = 0
            wrong_and_use_tool_and_direct_answer_num = 0
            only_use_tool_num = 0
            only_direct_answer_num = 0
            only_wrong_num = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                print(f"Data loading time: {time.time() - last_step_end_time}")
                timing_raw['data_loading'] = time.time() - last_step_end_time
                num_gen_batches += 1
                # pop those keys for generation
                if 'multi_modal_data' in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'video_files'],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    batch_rounds_num_list = []
                    for idx in range(len(new_batch.batch['multi_turn_response_mask'])):
                        prompt_and_response_ids = torch.cat([new_batch.batch['prompts'][idx], new_batch.batch['responses'][idx]], dim=0)
                        response_ids = prompt_and_response_ids[new_batch.batch['multi_turn_response_mask'][idx] == 1]
                        response = self.tokenizer.decode(response_ids)
                        
                        rounds_num = response.count("<|im_end|>")
                        batch_rounds_num_list.append(rounds_num)
                        round_nums_counter[rounds_num] += 1
                        assert rounds_num >= 1, f"response: {response}"
                        assert rounds_num <= self.config.actor_rollout_ref.rollout.max_generation_round, f"rounds_num {rounds_num} should not be greater than max_generation_round {self.config.actor_rollout_ref.rollout.max_generation_round}, response: {response}"
                    new_batch.batch['rounds_num'] = torch.tensor(batch_rounds_num_list, dtype=torch.int32)
                    # compute global_valid tokens
                    new_batch.meta_info['global_token_num'] = torch.sum(new_batch.batch['attention_mask'], dim=-1).tolist()

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # set tool_call_penalty
                        if self.config.trainer.get("float_penalty", False):
                            start_penalty = self.config.trainer['start_penalty']
                            end_penalty = self.config.trainer['end_penalty']
                            anneal_steps = self.config.trainer['anneal_steps']
                            tool_call_penalty = start_penalty + min(1.0, self.global_steps / anneal_steps) * (end_penalty - start_penalty)
                        else:
                            tool_call_penalty = self.config.reward_model['tool_call_penalty']
                        self.reward_fn.extra_info['tool_call_penalty'] = tool_call_penalty

                        # we combine with rule-based rm
                        from verl.workers.reward_manager import CustomRewardManager, NaiveMultiThreadsV2RewardManager, NaiveMultiThreadsRewardManager
                        if isinstance(self.reward_fn, CustomRewardManager):
                            reward_tensor, acc_reward_tensor, format_reward_tensor = self.reward_fn(new_batch)
                        elif isinstance(self.reward_fn, NaiveMultiThreadsV2RewardManager) or isinstance(self.reward_fn, NaiveMultiThreadsRewardManager):
                            reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids = self.reward_fn(new_batch)
                        else:
                            reward_tensor = self.reward_fn(new_batch)
                        new_batch.batch['token_level_scores'] = reward_tensor
                        if self.config.reward_model.log_rewards_separately:
                            if isinstance(self.reward_fn, CustomRewardManager) or isinstance(self.reward_fn, NaiveMultiThreadsV2RewardManager) or isinstance(self.reward_fn, NaiveMultiThreadsRewardManager):
                                new_batch.batch['format_scores'] = format_reward_tensor.sum(-1)
                                new_batch.batch['acc_scores'] = acc_reward_tensor.sum(-1)
                            else:
                                new_batch.batch['format_scores'] = self.reward_fn.cal_format_reward_for_logging(new_batch).sum(-1)
                                new_batch.batch['acc_scores'] = self.reward_fn.cal_acc_reward_for_logging(new_batch).sum(-1)
                        for idx in range(len(new_batch.batch['acc_scores'])):
                            rounds_num = new_batch.batch['rounds_num'][idx].item()
                            round_correct_counter[rounds_num] += (new_batch.batch['acc_scores'][idx].item() > 0)
                        invalid_num += len(invalid_uids)
                        solve_acc += new_batch.batch['acc_scores'].sum().item()
                        solve_format += new_batch.batch['format_scores'].sum().item()
                        uids = new_batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            if uid in invalid_uids:
                                valid_mask[uid_mask] = False
                                continue
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            uid_acc_rewards = acc_reward_tensor[uid_mask].sum(-1)
                            uid_format_rewards = format_reward_tensor[uid_mask].sum(-1)
                            if uid_rewards.std().item() < 1e-4:
                                valid_mask[uid_mask] = False
                            if (uid_rewards == 0).all():
                                solve_none += 1
                            elif (uid_acc_rewards == self.config.reward_model.acc_reward_weight).all() and (uid_format_rewards == self.config.reward_model.format_reward_weight).all():
                                solve_all += 1

                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_acc_rewards == 0).all():
                                solve_acc_none += 1
                            elif (uid_acc_rewards == self.config.reward_model.acc_reward_weight).all():
                                solve_acc_all += 1

                            if (uid_format_rewards == 0).all():
                                solve_format_none += 1
                            elif (uid_format_rewards == self.config.reward_model.format_reward_weight).all():
                                solve_format_all += 1
                            # Check if all rewards are 0 or all are 1 for this uid
                            if self.tool_call in ['video_zoom']:
                                uid_multi_modal_data = new_batch.non_tensor_batch['multi_modal_data'][uid_mask]
                                uid_prompt_img_num = new_batch.non_tensor_batch['prompt_img_num'][uid_mask]
                                success_tool_call_mask = torch.tensor([len(item['image']) > prompt_img_num for item, prompt_img_num in zip(uid_multi_modal_data, uid_prompt_img_num)])

                                success_tool_call_count = success_tool_call_mask.sum().item()
                                success_tool_call_per_batch += success_tool_call_count
                                success_tool_call_per_sample += (success_tool_call_count > 0)
                                # Compute Reward Distribution
                                use_tool_correct_answer_count = ((uid_acc_rewards > 0) & success_tool_call_mask).sum().item()
                                direct_answer_correct_answer_count = ((uid_acc_rewards > 0) & ~success_tool_call_mask).sum().item()
                                wrong_answer_count = (uid_acc_rewards == 0).sum().item()
                                if use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count == 0:
                                    use_tool_and_direct_answer_num += 1
                                elif use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count == 0 and wrong_answer_count > 0:
                                    wrong_and_use_tool_num += 1
                                elif use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count == 0 and wrong_answer_count == 0:
                                    only_use_tool_num += 1
                                elif use_tool_correct_answer_count == 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count > 0:
                                    wrong_and_direct_answer_num += 1
                                elif use_tool_correct_answer_count == 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count == 0:
                                    only_direct_answer_num += 1
                                elif use_tool_correct_answer_count == 0 and direct_answer_correct_answer_count == 0 and wrong_answer_count > 0:
                                    only_wrong_num += 1
                                elif use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count > 0:
                                    wrong_and_use_tool_and_direct_answer_num += 1
                                else:
                                    raise NotImplementedError(f"use_tool_correct_answer_count: {use_tool_correct_answer_count}, direct_answer_correct_answer_count: {direct_answer_correct_answer_count}, wrong_answer_count: {wrong_answer_count}.")

                            totals += 1
                        effective_num += valid_mask.sum().item() // self.config.actor_rollout_ref.rollout.n
                        if self.config.trainer.rejection_sample:
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue
                            # Filter batch to keep only valid samples
                            new_batch = new_batch[valid_mask]
                            new_batch = dataprotoitem_to_dataproto(new_batch)

                            assert new_batch.batch['input_ids'].shape[0] % self.config.actor_rollout_ref.rollout.n == 0
                            num_prompt_in_batch += new_batch.batch['input_ids'].shape[0] // self.config.actor_rollout_ref.rollout.n

                            if batch is None:
                                batch = new_batch
                            else:
                                batch = DataProto.concat([batch, new_batch])

                            prompt_bsz = self.config.data.train_batch_size
                            if num_prompt_in_batch < prompt_bsz:
                                print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                                max_num_gen_batches = self.config.algorithm.max_num_gen_batches
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f'{num_gen_batches=}. Keep generating...')
                                    continue
                                else:
                                    raise ValueError(
                                        f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                    )
                            else:
                                # Align the batch
                                traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n

                                batch = batch[:traj_bsz]

                                batch = dataprotoitem_to_dataproto(batch)

                        else:
                            batch = new_batch

                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        self._balance_batch(batch, metrics=metrics)
                        metrics['batch/solve_acc'] = solve_acc / (totals * self.config.actor_rollout_ref.rollout.n)
                        metrics['batch/solve_format'] = solve_format / (totals * self.config.actor_rollout_ref.rollout.n)
                        metrics['batch/solve_none'] = solve_none / totals
                        metrics['batch/solve_all'] = solve_all / totals
                        metrics['batch/solve_acc_none'] = solve_acc_none / totals
                        metrics['batch/solve_acc_all'] = solve_acc_all / totals
                        metrics['batch/solve_format_none'] = solve_format_none / totals
                        metrics['batch/solve_format_all'] = solve_format_all / totals
                        metrics['batch/invalid_ratio'] = invalid_num / totals
                        metrics['batch/effective_ratio'] = effective_num / totals
                        metrics['batch/success_tool_call_per_batch'] = success_tool_call_per_batch / (totals * self.config.actor_rollout_ref.rollout.n)
                        metrics['batch/success_tool_call_per_sample'] = success_tool_call_per_sample / totals
                        metrics['batch/tool_call_penalty'] = tool_call_penalty
                        for idx in range(self.config.actor_rollout_ref.rollout.max_generation_round):
                            metrics[f'round_distribution/round_{idx+1}_ratio'] = round_nums_counter[idx+1] / sum(round_nums_counter.values())
                            metrics[f'round_distribution/round_{idx+1}_acc_ratio'] = round_correct_counter[idx+1] / (round_nums_counter[idx+1] + 1e-5)
                        if self.tool_call in ['video_zoom']:
                            metrics['original_reward_distribution/only_wrong_answer_ratio'] = only_wrong_num / totals
                            metrics['original_reward_distribution/only_correct_direct_answer_ratio'] = only_direct_answer_num / totals
                            metrics['original_reward_distribution/only_correct_use_tool_ratio'] = only_use_tool_num / totals
                            metrics['original_reward_distribution/correct_use_tool_and_direct_answer_ratio'] = use_tool_and_direct_answer_num / totals
                            metrics['original_reward_distribution/wrong_answer_and_correct_use_tool_ratio'] = wrong_and_use_tool_num / totals
                            metrics['original_reward_distribution/wrong_answer_and_correc_direct_answer_ratio'] = wrong_and_direct_answer_num / totals
                            metrics['original_reward_distribution/wrong_answer_and_correct_use_tool_and_direct_answer_ratio'] = wrong_and_use_tool_and_direct_answer_num / totals
                            batch.batch["success_tool_call"] = torch.tensor([len(item['image']) > prompt_img_num for item, prompt_img_num in zip(batch.non_tensor_batch['multi_modal_data'], batch.non_tensor_batch['prompt_img_num'])])
                            batch.batch["tool_call_frame_num"] = [len(item['image']) - prompt_img_num for item, prompt_img_num in zip(batch.non_tensor_batch['multi_modal_data'], batch.non_tensor_batch['prompt_img_num'])]

                            metrics = self._maybe_log_tool_call_metrics(batch, metrics)

                        self.reward_fn.extra_info['success_tool_call_rate'] = success_tool_call_per_batch / (totals * self.config.actor_rollout_ref.rollout.n)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # log training rollouts
                    if self.config.trainer.log_training_rollouts_freq > 0 and self.global_steps % self.config.trainer.log_training_rollouts_freq == 0:
                        self._maybe_log_train_generations_to_wandb(batch, batch.batch['token_level_scores'])

                    # log reflection frequency
                    if self.config.trainer.reflection_keywords is not None:
                        for keyword in self.config.trainer.reflection_keywords:
                            pattern = re.compile(keyword, re.IGNORECASE)
                            metrics.update({
                                f'reflection_freq/{keyword}':
                                torch.mean(torch.tensor([
                                    len(pattern.findall(self.tokenizer.decode(data_item.batch['responses'], skip_special_tokens=True))) 
                                    for data_item in batch
                                ], dtype=torch.float)).detach().item()
                            })
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            response_length = batch.batch['responses'].shape[-1]

                            prompt_mask = batch.batch['attention_mask'][:, :-response_length]
                            response_mask = batch.batch['attention_mask'][:, -response_length:]
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                dump_path=rollout_data_dir,
                                reward_extra_infos_dict={
                                    "tool_call": batch.batch["success_tool_call"].cpu().tolist(),
                                    "acc_score": batch.batch["acc_scores"].cpu().tolist(),
                                    "format_score": batch.batch["format_scores"].cpu().tolist(),
                                    "prompt_len": prompt_mask.sum(-1).float().cpu().tolist(),
                                    "response_len": response_mask.sum(-1).float().cpu().tolist(),
                                }
                            )
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, max_response_length=self.config.data.max_response_length))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                solve_acc = 0
                solve_format = 0
                solve_none = 0
                solve_all = 0
                solve_acc_none = 0
                solve_acc_all = 0
                solve_format_none = 0
                solve_format_all = 0
                success_tool_call_per_batch = 0
                success_tool_call_per_sample = 0
                totals = 0
                invalid_num = 0
                effective_num = 0
                round_nums_counter = defaultdict(int)
                round_correct_counter = defaultdict(int)
                if self.tool_call in ['video_zoom']:
                    use_tool_and_direct_answer_num = 0
                    wrong_and_use_tool_num = 0
                    wrong_and_direct_answer_num = 0
                    wrong_and_use_tool_and_direct_answer_num = 0
                    only_use_tool_num = 0
                    only_direct_answer_num = 0
                    only_wrong_num = 0
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                last_step_end_time = time.time()
        if self.config.trainer.save_freq > 0:
            with _timer('save_checkpoint', timing_raw):
                self._save_checkpoint()
        return

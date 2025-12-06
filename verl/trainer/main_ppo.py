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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
        #from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    if config.actor_rollout_ref.actor.use_kl_loss and config.actor_rollout_ref.actor.kl_loss_coef > 0:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    elif reward_manager_name.startswith('custom'):
        from verl.workers.reward_manager import CustomRewardManager
        reward_manager_cls = CustomRewardManager
        compute_score = reward_manager_name.split('@')[-1]
        # Note that we always use function-based RM for validation
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, mode="train", acc_reward_weight=config.reward_model.get("acc_reward_weight", 0.9), format_reward_weight=config.reward_model.get("format_reward_weight", 0.1), overlength_reward_weight=config.reward_model.get("overlength_reward_weight", 0.0))
        # assert config.reward_model.log_rewards_separately == False, "Custom reward manager does not support log_rewards_separately"
    elif reward_manager_name == 'naive_multithreads':
        from verl.workers.reward_manager import NaiveMultiThreadsRewardManager
        reward_manager_cls = NaiveMultiThreadsRewardManager
        extra_info = {
            "acc_reward_weight": config.reward_model.get("acc_reward_weight", 0.9),
            "format_reward_weight": config.reward_model.get("format_reward_weight", 0.1),
            "tool_call_penalty": config.reward_model.get("tool_call_penalty", 0.0),
            "forced_tool_call": config.reward_model.get("forced_tool_call", False),
            "gpt_extract_answer": config.reward_model.get("gpt_extract_answer", False),
            "penalize_correct_tool_call": config.reward_model.get("penalize_correct_tool_call", True),
            "penalize_incorrect_tool_call": config.reward_model.get("penalize_incorrect_tool_call", False),
            "penalty_per_turn": config.reward_model.get("penalty_per_turn", 0.0),
            "target_tool_call_rate": config.reward_model.get("target_tool_call_rate", 1.0),
            "judge_mc_by_gpt": config.reward_model.get("judge_mc_by_gpt", False),
            "tool_call_max_frames": config.actor_rollout_ref.rollout.get("tool_call_max_frames", 16),
            "filter_unfinished_traj": config.reward_model.get("filter_unfinished_traj", False),
            "tool_call_frame_target": config.reward_model.get("tool_call_frame_target", 0),
            "tool_call_frame_num_penalty": config.reward_model.get("tool_call_frame_num_penalty", 0.0),
        }
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, mode="train", extra_info=extra_info)
    elif reward_manager_name == 'naive_multithreads_v2':
        from verl.workers.reward_manager import NaiveMultiThreadsV2RewardManager
        reward_manager_cls = NaiveMultiThreadsV2RewardManager
        extra_info = {
            "acc_reward_weight": config.reward_model.get("acc_reward_weight", 0.9),
            "format_reward_weight": config.reward_model.get("format_reward_weight", 0.1),
        }
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, mode="train", extra_info=extra_info)
    else:
        raise NotImplementedError

    val_reward_manager_name = config.reward_model.get("val_reward_manager", None)
    if val_reward_manager_name is None:
        val_reward_manager_name = reward_manager_name
    if val_reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)
    elif val_reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)
    elif val_reward_manager_name.startswith('custom'):
        from verl.workers.reward_manager import CustomRewardManager
        reward_manager_cls = CustomRewardManager
        compute_score = val_reward_manager_name.split('@')[-1]
        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, mode="val")
        # assert config.reward_model.log_rewards_separately == False, "Custom reward manager does not support log_rewards_separately"
    elif reward_manager_name == 'naive_multithreads':
        from verl.workers.reward_manager import NaiveMultiThreadsRewardManager
        reward_manager_cls = NaiveMultiThreadsRewardManager
        extra_info = {
            "acc_reward_weight": 1.0,
            "format_reward_weight": 1.0,
            "tool_call_penalty": 0,
            "penalty_per_turn": 0,
            "forced_tool_call": False,
            "gpt_extract_answer": config.reward_model.get("gpt_extract_answer", False),
            "penalize_correct_tool_call": False,
            "penalize_incorrect_tool_call": False,
            "judge_mc_by_gpt": config.reward_model.get("judge_mc_by_gpt", False),
            "tool_call_max_frames": config.actor_rollout_ref.rollout.get("tool_call_max_frames", 16),
            "filter_unfinished_traj": False,
        }
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, mode="val", extra_info=extra_info)
    elif reward_manager_name == 'naive_multithreads_v2':
        from verl.workers.reward_manager import NaiveMultiThreadsV2RewardManager
        reward_manager_cls = NaiveMultiThreadsV2RewardManager
        extra_info = {
            "acc_reward_weight": 1.0,
            "format_reward_weight": 0.0,
        }
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, mode="val", extra_info=extra_info)
    else:
        raise NotImplementedError
    
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            processor=processor,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()

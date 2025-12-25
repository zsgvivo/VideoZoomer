import os
import re
import time
import io
import asyncio
import logging
import json
from collections.abc import AsyncGenerator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import cloudpickle
import pickle
import zmq
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length, get_eos_mask, get_final_eos_mask, pad_2d_list_to_length
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.workers.rollout.vllm_rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.vllm_rollout.vllm_rollout import _pre_process_inputs
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase
from .function_tools import prepare_tool_call_inputs, extract_video_clip, prepare_tool_call_inputs_video
from PIL import Image

def _get_model_runner_workers(vllm_config, init_ray: bool = True):
    assert vllm_config.instance_id is not None, "instance_id must be set for external ray actors."
    fields = vllm_config.instance_id.split(":")
    assert len(fields) == 4, (
        f"instance_id: {vllm_config.instance_id} must be in the format of "
        f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
    )
    namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])
    # Make sure subprocess in same namespace as parent actor.
    # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
    if init_ray:
        ray.init(namespace=namespace)
    actor_names = [
        actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")
    ]
    vllm_tp_size = vllm_config.parallel_config.tensor_parallel_size
    assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
        f"instance_id: {vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: "
        f"{vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."
    )
    def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
        fields = actor_name.split(":")
        assert len(fields) == 2, f"invalid actor name: {actor_name}"
        pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
        return pg_index, local_rank
    # sort actor names by pg_index and local_rank
    actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
    actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
    workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
    print(f"instance_id: {vllm_config.instance_id} initializes with external actors: {actor_names}")
    return workers

def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(torch.cat([t,torch.tensor([pad_token_id] * (max_len-t.size(0)), device=t.device, dtype=t.dtype)],dim=0))
    return torch.stack(padded_tensor_list, dim=dim)

class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."
        self.workers = _get_model_runner_workers(vllm_config=self.vllm_config, init_ray=True)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} intializes finished.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        outputs = ray.get(
            [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers]
        )
        return outputs

    def check_health(self):
        return

class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))
        return outputs

    def check_health(self):
        return



def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


@ray.remote(num_cpus=1)
class AsyncvLLMEngine:
    """
    AsyncvLLMEngine is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    2. AsyncLLM spawn EngineCore in subprocess.
    3. EngineCore initialize ExternalRayDistributedExecutor.
    4. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    5. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str, tokenizer, processor):
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        self.config = config
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.tokenizer = tokenizer
        self.engine: AsyncLLM = None
        self.pad_token_id = self.tokenizer.pad_token_id

        self.max_generation_round = self.config.rollout.get('max_generation_round', 1)

        self.processor = processor
        self.merge_length = self.processor.image_processor.merge_size ** 2


    def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.max_total_response_length #config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        kwargs['n'] = 1

        print(f"override_generation_config: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "ray")
        if backend == "zeromq":
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        elif backend == "ray":
            distributed_executor_backend = ExternalRayDistributedExecutor
        else:
            distributed_executor_backend = None

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            # disable_log_stats=config.disable_log_stats,
            disable_log_stats=False,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
            limit_mm_per_prompt=config.get('limit_mm_per_prompt', {'image': 1}),
        )

        # init async llm engine
        vllm_config = self._create_engine_config(engine_args)
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

    def _create_engine_config(self, engine_args: AsyncEngineArgs):
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"

        # VERL_VLLM_ZMQ_ADDRESSES
        if engine_args.distributed_executor_backend == ExternalZeroMQDistributedExecutor:
            workers = _get_model_runner_workers(vllm_config=vllm_config, init_ray=False)
            zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in workers])
            print(f"VERL_VLLM_ZMQ_ADDRESSES: {zmq_addresses}")
            os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        return vllm_config

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n):
        req_list = []
        for data_idx, _raw_prompt_id in enumerate(prompts.non_tensor_batch["raw_prompt_ids"]):
            _multimodal_data = prompts.non_tensor_batch["multi_modal_data"][data_idx]
            _video_path = prompts.non_tensor_batch["video_files"][data_idx][0]
            _input_ids = prompts.batch['input_ids'][data_idx]
            _attention_mask = prompts.batch['attention_mask'][data_idx]
            _position_ids = prompts.batch['position_ids'][data_idx]

            for idx in range(n):
                
                req = AsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=idx,
                    request_id=str(uuid4()),
                    multi_modal_data=_multimodal_data,
                    raw_prompt_id=_raw_prompt_id,
                    reward_scores={},
                    max_response_len=self.config.rollout.response_length,
                    max_model_len=(
                        self.config.rollout.max_model_len
                        or self.config.rollout.prompt_length + self.config.rollout.response_length
                    ),
                    video_path=_video_path,
                    input_ids=_input_ids,
                    attention_mask=_attention_mask,
                    position_ids=_position_ids
                )
                req_list.append(req)

        return req_list

    def post_process_single(self, prefix_length,prompt_img_nums, vllm_input, multi_turn_response_mask):

        # for each traj, we skip first-round prompt_ids/attention_mask
        first_round_prompt_length = prefix_length
        # Repeat <|image_pad|> token id for modeling_qwen2vl
        generation_response_ids = vllm_input['prompt_token_ids'][first_round_prompt_length:]

        if len(vllm_input['multi_modal_data']['image']) > prompt_img_nums:
            new_image_inputs = self.processor.image_processor(vllm_input['multi_modal_data']['image'][prompt_img_nums:], return_tensors='pt')    # NOTE: The fisrt image is the original image, here we only take the resized image into account
            image_grid_thws = new_image_inputs['image_grid_thw']
    
        all_response_masks = torch.cat(multi_turn_response_mask[1:], dim=0).tolist()
        
        assert len(generation_response_ids) == len(all_response_masks)
        index, image_pad_token, magic_num = 0, 151655, 654321
        while image_pad_token in generation_response_ids:
            image_pad_token_pos = generation_response_ids.index(image_pad_token)
            image_pad_token_repeat_num = image_grid_thws[index].prod() // self.merge_length
            # update response_tensor_ids
            generation_response_ids[image_pad_token_pos : image_pad_token_pos + 1] = [magic_num] * image_pad_token_repeat_num
            # update all_response_masks
            all_response_masks[image_pad_token_pos : image_pad_token_pos + 1] = [0] * image_pad_token_repeat_num
            index += 1
        if index > 0:
            assert index == len(image_grid_thws)
        generation_response_ids = [image_pad_token if x == magic_num else x for x in generation_response_ids]
        all_response = torch.tensor(generation_response_ids, dtype=torch.int64, device=multi_turn_response_mask[0].device)
        all_response_masks = torch.tensor(all_response_masks, dtype=torch.int64, device=multi_turn_response_mask[0].device)

        return all_response, all_response_masks

    def process_tool_call(self, vllm_input,tool_call_contents, decoded_resp_, multi_turn_response_mask, current_iteration, max_iterations, video_path):
        error_info = None
        try:
            json_pattern = re.compile(r'\{.*?\}')
            json_objects = []
            for content in tool_call_contents:
                json_strings = json_pattern.findall(content)
                json_objects.extend([json.loads(json_str) for json_str in json_strings])
            tool_info = prepare_tool_call_inputs_video(json_objects)
        
        except Exception as e:
            print(str(e))
            error_info = str(e)
            tool_type = None
            args = None
        
        if error_info is not None:
            tool_outputs = f"<tool_response>\nERROR occurs during the function tool call. Error Information: {error_info}.\n</tool_response>\n"
        else:
            tool_outputs = extract_video_clip(
                video_path=video_path,
                **tool_info,
                storage_system='local',
                max_pixels=self.config.rollout.data_max_pixels,
                min_pixels=self.config.rollout.data_min_pixels,
                max_frames=self.config.rollout.tool_call_max_frames,
            )
        
        if not isinstance(tool_outputs, str):
            # Construct Next Round Prompt
            frame_time = tool_outputs['frame_time']
            vision_pad = ''.join([f'<frame{i}_time{frame_time[i]:.2f}s><|vision_start|><|image_pad|><|vision_end|>' for i in range(len(frame_time))])
            tool_call_prompt_message = "<|im_start|>user\n" + "<tool_response>\nThe frames of the video clip are shown below:\n"+ vision_pad + "\n</tool_response>\n" + "continue your reasoning process inside <think> and </think> and then write your final answer inside <answer> and </answer>"
            if current_iteration == max_iterations - 2:
                tool_call_prompt_message += "Do not call <video_zoom> in this round, give a final answer based on information above."
            tool_call_prompt_message +=  "<|im_end|>\n<|im_start|>assistant\n"
            next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
            # update conversation
            vllm_input['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'

            vllm_input['multi_modal_data']['image'].extend(tool_outputs['frames'])
            multi_turn_response_mask.append(torch.zeros(len(next_turn_prompt_ids), dtype=multi_turn_response_mask[-1].dtype, device=multi_turn_response_mask[-1].device)) # USER, Mark as 0

        else:
            tool_call_prompt_message = "<|im_start|>user\n" + tool_outputs + "Please analyze the error information obtained from the function tool and adjust your response. Countinue your reasoning process inside <think> and </think>."
            if current_iteration == max_iterations - 2:
                tool_call_prompt_message += "Do not call <video_zoom> in this round, give a final answer based on information above."
            tool_call_prompt_message +=  "<|im_end|>\n<|im_start|>assistant\n"

            next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
            vllm_input['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
            multi_turn_response_mask.append(torch.zeros(len(next_turn_prompt_ids), dtype=multi_turn_response_mask[-1].dtype, device=multi_turn_response_mask[-1].device)) # USER, Mark as 0

        return 0

    async def _async_rollout_a_request(
        self, req: AsyncRolloutRequest, semaphore: asyncio.Semaphore, do_sample: bool = True, is_validate: bool = False, **kwargs
    ) -> AsyncRolloutRequest:
        async with semaphore:
            loop = asyncio.get_running_loop()
            _req = await loop.run_in_executor(None, lambda: deepcopy(req))
            finish_reason_type = None
            output = None
            current_turns = 0

            kwargs = {
                "n": 1,
                "stop": ['</video_zoom>',],
                "detokenize": True,
                "include_stop_str_in_output": True,
            }
            if not do_sample:
                kwargs.update({
                    "best_of": 1,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "temperature": 0
                })
            
            vllm_input = {}
            if _req.multi_modal_data:

                multi_modal_data = _req.multi_modal_data
                vllm_input = {'prompt_token_ids': _req.raw_prompt_id, 'multi_modal_data': multi_modal_data}

            prefix_length = len(_req.raw_prompt_id)
            prompt_img_nums = len(_req.multi_modal_data['image'])

            input_ids = torch.tensor(_req.input_ids).unsqueeze(0)
            attention_mask = torch.tensor(_req.attention_mask).unsqueeze(0)
            position_ids = torch.tensor(_req.position_ids).unsqueeze(0)

            multi_turn_response_mask = [torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)]

            n = self.config.rollout.val_n if is_validate else self.config.rollout.n  # TODO: for validate, do_sample=False

            
            max_image_num = self.config.rollout.get('limit_mm_per_prompt', {'image': 1})['image']
            max_iterations = self.max_generation_round
            
            pattern = re.compile(r'<video_zoom>(.*?)</video_zoom>', re.DOTALL)
            current_iteration = 0
            exceed = False
            while current_iteration < max_iterations:
                sampling_params = deepcopy(self.sampling_params)
                for key, value in kwargs.items():
                    if hasattr(sampling_params, key):
                        setattr(sampling_params, key, value)
                        
                outputs = self.engine.generate(
                    prompt=vllm_input,  # because we have already convert it to prompt token id
                    sampling_params=sampling_params,
                    request_id=_req.request_id+str(current_iteration),
                )

                async for res in outputs:
                    results = res

                content = results.outputs[0].text

                _token_ids = results.outputs[0].token_ids
                filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
                if 151645 not in filtered_token_ids:
                    filtered_token_ids = filtered_token_ids + [151645,]
                response_ = filtered_token_ids
                vllm_input['prompt_token_ids'] += response_
                multi_turn_response_mask.append(torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)) # ASSISTANT, Mark as 1

                decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
                tool_call_contents = pattern.findall(decoded_resp_)

                if len(tool_call_contents) > 0:
                    if (len(vllm_input['multi_modal_data']['image']) >= max_image_num) or (current_iteration == max_iterations - 1):   # If the current traj has already reached max_image_num, but still try to call tool, we should remove this traj.
                        exceed = True
                        break

                    new_context_length = await loop.run_in_executor(None, lambda: self.process_tool_call(vllm_input,tool_call_contents, decoded_resp_, multi_turn_response_mask, current_iteration, max_iterations, req.video_path))

                else:
                    break
                
                current_iteration += 1

            finish_reason = results.outputs[0].finish_reason
            finish_reason_type = FinishReasonTypeEnum.from_str(finish_reason)

            avg_response_tokens_per_turn = torch.cat(multi_turn_response_mask, dim=0).sum(-1).item() / (current_iteration + 1)

            all_response, all_response_masks = self.post_process_single(prefix_length, prompt_img_nums, vllm_input, multi_turn_response_mask)

            _req.all_response_ids = all_response
            _req.all_response_masks = all_response_masks
            _req.multi_modal_data = vllm_input['multi_modal_data']
            _req.exceed = exceed
            _req.avg_response_tokens_per_turn = avg_response_tokens_per_turn

            return _req

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    async def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device
        semaphore = asyncio.Semaphore(self.config.rollout.max_concurrency)

        req_list = self._preprocess_prompt_to_async_rollout_requests(
            prompts,
            n = self.config.rollout.val_n if is_validate else self.config.rollout.n
        )
        t0 = time.time()
        with torch.no_grad():
            output_req_list = await asyncio.gather(
                *[self._async_rollout_a_request(req, semaphore, do_sample, is_validate, **kwargs) for req in req_list]
            )
        t1 = time.time()
        print(f"time: {t1-t0}")
        sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))


        return self.post_process(prompts, sorted_output_req_list)

    def post_process(self, prompts: DataProto, output_req_list: List[AsyncRolloutRequest]) -> DataProto:
        config = self.config.rollout
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        # convert to DataProto
        response = []
        response_generation_mask =[]
        multi_modal_data_list = []
        raw_prompts = []
        uid_list = []
        data_source_list = []
        ground_truth_list = []
        exceed_list = []
        avg_response_tokens_per_turn_list = []

        reward_tensor_list = []
        acc_reward_tensor_list = []
        format_reward_tensor_list = []
        overlong_reward_tensor_list = []
        invalid_uids = []
        
        for req in output_req_list:
            response.append(req.all_response_ids)
            response_generation_mask.append(req.all_response_masks) # at least we have single-turn conversation
            multi_modal_data_list.append(req.multi_modal_data)
            exceed_list.append(req.exceed)
            avg_response_tokens_per_turn_list.append(req.avg_response_tokens_per_turn)
        
        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(response, self.pad_token_id, dim=0) # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0) # Tensor, (B'*R, padded_length)
        assert all([response.size(dim) == response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.max_total_response_length)
        if response.shape[1] > config.max_total_response_length:
            response = response[:,:config.max_total_response_length]
            response_generation_mask = response_generation_mask[:,:config.max_total_response_length]
        elif response.shape[1] < config.max_total_response_length:
            response = pad_sequence_to_length(response, config.max_total_response_length, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, config.max_total_response_length, 0)

        prompt_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch['attention_mask']  # (bs*tp, max_prompt_length), left padding 0
        position_ids = prompts.batch['position_ids']  # (bs*tp, max_prompt_length), left padding 0

        n = config.val_n if is_validate else config.n
        if n > 1 and do_sample:
            prompt_ids = _repeat_interleave(prompt_ids, n) # (B, max_prompt_length) -> (B*R, max_prompt_length)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)

        # FIXME: function get_final_eos_mask cannot handle cases that when there is no <|im_end|> in the given response
        # response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, <|im_end|> is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, |im_end| is 151645

        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        input_prompt_generation_mask = torch.zeros_like(prompt_ids, dtype=attention_mask.dtype, device=attention_mask.device) # (B'*R, max_prompt_length), all 0
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        seq = torch.cat([prompt_ids, response], dim=-1) # (B*R, max_prompt_length + max_total_response_length)

        # NOTE: We repeat 'multi_modal_data'
        non_tensor_batch = {}
        non_tensor_batch['multi_modal_data'] = np.array(multi_modal_data_list)
        non_tensor_batch['avg_response_tokens_per_turn'] = np.array(avg_response_tokens_per_turn_list, dtype=object)

        response_length = response.size(1)
        batch_size = prompt_ids.shape[0]
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids_list = []
            for prompt_with_response, attn_mask, multi_modal_data in zip(seq, attention_mask, multi_modal_data_list):
                image_inputs = self.processor.image_processor(images=multi_modal_data['image'],videos=multi_modal_data['video'], return_tensors='pt')
                image_grid_thw = image_inputs['image_grid_thw']
                video_grid_thw = image_inputs["video_grid_thw"]
                pos_ids = get_rope_index(
                    self.processor,
                    input_ids=prompt_with_response,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attn_mask,
                )
                position_ids_list.append(pos_ids)
            position_ids = torch.stack(position_ids_list, dim=0)
        else:
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            # TODO(sgm): fix position_ids on right_pad
            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        batch = TensorDict(
            {
                'prompts': prompt_ids.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous(),
            },
            batch_size=batch_size
        )

        data = DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch
        )

        return data

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.engine.generate(request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def wake_up(self):
        await self.engine.wake_up()

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()

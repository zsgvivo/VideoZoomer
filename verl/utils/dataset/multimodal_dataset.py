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

import os
from omegaconf import ListConfig
from collections import defaultdict
from decord import VideoReader, cpu
from typing import Any, List, Union, Optional
import copy
import pandas as pd
import yaml
import json
import math
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from tqdm import tqdm
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.trainer.constants import OPEN_ENDED_DATA_SOURCES
# from verl.utils.dataset.tt.utils.video_player import Client

### IMPORTANT ###
import datasets
from datasets import load_from_disk, concatenate_datasets
datasets.disable_caching()

RULE_BASED_QUESTION_TEMPLATE = "{Question}{post_prompt}"
OPEN_ENDED_QUESTION_TEMPLATE = "{Question}{post_prompt}"

def make_conversation_multimodal(sources, system_prompt=None, post_prompt=None):
    for source in sources:
        data_source = source['data_source']
        question_template = OPEN_ENDED_QUESTION_TEMPLATE if data_source in OPEN_ENDED_DATA_SOURCES else RULE_BASED_QUESTION_TEMPLATE
        if post_prompt is not None:
            problem = question_template.format(Question=source["problem"], post_prompt=post_prompt)
        else:
            problem = source["problem"]
        prompt = [
            {
                "role": "user",
                "content": problem,
            },
        ]
        if system_prompt is not None:
            prompt.insert(0, {"role": "system", "content": system_prompt})
        source["prompt"] = prompt
    return sources

def make_conversation_multimodal_hf(sample, system_prompt=None, post_prompt=None):
    data_source = sample['data_source']
    question_template = OPEN_ENDED_QUESTION_TEMPLATE if data_source in OPEN_ENDED_DATA_SOURCES else RULE_BASED_QUESTION_TEMPLATE
    problem = question_template.format(Question=sample["problem"], post_prompt=post_prompt)
    prompt = [
        {
            "role": "user",
            "content": problem,
        },
    ]
    if system_prompt is not None:
        prompt.insert(0, {"role": "system", "content": system_prompt})
    sample["prompt"] = prompt
    return sample

def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}

def process_hf_dataset(dataset, system_prompt=None, post_prompt=None):
    # currently for image pretrain only, since volume is big
    # we resort to efficient hf dataset loading
    cur_data_path = dataset.get("data_path")
    cur_ds = load_from_disk(cur_data_path)

    sampling_strategy = dataset.get("sampling_strategy", "all")
    sampling_number = None
    if ":" in sampling_strategy:
        sampling_strategy, sampling_number = sampling_strategy.split(":")
        if "%" in sampling_number:
            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_ds) / 100)
        else:
            # sampling_number = int(sampling_number)
            raise NotImplementedError()
    # Apply the sampling strategy
    if sampling_strategy == "first" and sampling_number is not None:
        cur_ds = cur_ds.select(range(sampling_number))
    elif sampling_strategy == "end" and sampling_number is not None:
        # cur_data_dict = cur_data_dict[-sampling_number:]
        raise NotImplementedError()
    elif sampling_strategy == "random" and sampling_number is not None:
        # random.shuffle(cur_data_dict)
        # cur_data_dict = cur_data_dict[:sampling_number]
        raise NotImplementedError()
    cur_ds = cur_ds.map(make_conversation_multimodal_hf, fn_kwargs={'system_prompt': system_prompt, 'post_prompt': post_prompt}, num_proc=64)   # we have 8 num_workers
    # image pretrain only need these columns, remove other column for efficiency
    cur_ds = cur_ds.remove_columns(
        [col for col in cur_ds.column_names if col not in ['doc_id', 'images', 'problem', 'solution', 'prompt', 'data_source', 'videos']]
    )
    print(f"Loaded {cur_ds.num_rows} samples from {cur_data_path}.")
    return cur_ds

class MultiModalDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=True,
                 truncation='error',
                 system_prompt: str = None,
                 post_prompt: str = None,
                 max_pixels: int = 100352,
                 min_pixels: int = 100352 / 4,
                 video_fps: int = 0.2,
                 is_fix_frame: bool = False,
                 frames_upbound: int = 30,
                 is_qwen: bool = True,
                 pass_video_as_frames: bool = True,
                 fast_seek: bool = False,
                 random_fps: dict = None,
                ):

        self.data_path = copy.deepcopy(data_path)
        self.original_parquet_files = copy.deepcopy(data_path)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        if system_prompt and system_prompt.endswith('.txt'):
            with open(system_prompt, 'r') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = system_prompt
        
        if post_prompt and post_prompt.endswith('.txt'):
            with open(post_prompt, 'r') as f:
                self.post_prompt = f.read()
        else:
            self.post_prompt = post_prompt
        # Image Arguments
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        # Video Arguments
        self.video_fps = video_fps
        self.is_fix_frame = is_fix_frame
        self.frames_upbound = frames_upbound
        # Define Model Type
        self.is_qwen = is_qwen
        self.pass_video_as_frames = pass_video_as_frames
        self.fast_seek = fast_seek
        self.random_fps = random_fps
        random.seed(42)

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        # self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        list_data_dict = []
        dataset_format = ""  # check whether hf ds is used
        if self.data_path.endswith(".yaml"):
            with open(self.data_path) as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                for dataset in datasets:
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None
                    if "data_path" in dataset:
                        dataset_format = 'hf'
                        cur_data_dict = process_hf_dataset(dataset, system_prompt=self.system_prompt, post_prompt=self.post_prompt)
                        list_data_dict.append(cur_data_dict)
                        continue

                    json_path = dataset.get("json_path")
                    local_json_path = json_path
                    with open(local_json_path) as json_file:
                        cur_data_dict = json.load(json_file)

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    list_data_dict.extend(make_conversation_multimodal(cur_data_dict, self.system_prompt, self.post_prompt))
        elif self.data_path.endswith(".json"):
            json_data = json.load(open(self.data_path, 'r'))
            list_data_dict.extend(make_conversation_multimodal(json_data, self.system_prompt, self.post_prompt))
        else:
            raise ValueError('The data_path file should be a YAML file.')
        
        if dataset_format == 'hf':
            list_data_dict = concatenate_datasets(list_data_dict)

        all_keys = set()
        for data_point in list_data_dict:
            all_keys.update(data_point.keys())
        
        # Ensure each data point has all keys, filling missing ones with None
        for data_point in list_data_dict:
            for key in all_keys:
                if key not in data_point:
                    data_point[key] = None

        print(f">>> Overall Data Size: {len(list_data_dict)}")
        self.list_data_dict = list_data_dict

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_data_path') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def process_image(self, image_file: Any, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512, storage_system: str = 'local'):
        from io import BytesIO
        from PIL import Image

        if isinstance(image_file, dict):
            image = Image.open(BytesIO(image_file['bytes']))
        elif isinstance(image_file, np.ndarray):
            image = Image.fromarray(image_file).convert('RGB')
        elif isinstance(image_file, Image.Image):
            image = image_file.convert('RGB')
        elif isinstance(image_file, str):
            if not os.path.exists(image_file):
                raise FileNotFoundError(f"Image path {image_file} does not exist.")
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as exn:
                print(f"Failed to open image {image_file}. Exception:", exn)
                raise exn
        else:
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as exn:
                print(f"Failed to open image {image_file}. Exception:", exn)
                raise exn

        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    def process_video(self, video_id_or_path: str, video_fps: float, frames_upbound: int, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512, storage_system: str = 'local', is_accurate: bool = True, preprocess=True):
        from io import BytesIO
        import cv2
        from PIL import Image
        import numpy as np
        import time
        import uuid
        from concurrent.futures import ThreadPoolExecutor
        try:
            start_time = time.time()
            # import pdb; pdb.set_trace()
            # Currently, we do not support load precomputed features
            if preprocess:
                cache_path = f'/mnt/bn/tiktok-mm-5/aiic/users/dingyang/data/preprocessed_frames/fps{video_fps}_upbound{frames_upbound}_maxpixels{max_pixels}_minpixels{min_pixels}/{uuid.uuid5(uuid.NAMESPACE_DNS, video_id_or_path)}'
                def load_frame(i):
                    frame = Image.open(os.path.join(cache_path, f'frame{i}.png'))
                    frame.verify()
                    frame.close()
                    return Image.open(os.path.join(cache_path, f'frame{i}.png'))
                try:
                    if os.path.exists(cache_path):
                        frame_time = np.loadtxt(os.path.join(cache_path, 'frame_time.txt'))
                        with ThreadPoolExecutor(max_workers=8) as executor:
                            frames = list(executor.map(load_frame, range(len(frame_time))))
                        print(f'video {video_id_or_path} is preprocessed, loaded in {time.time() - start_time:.2f}s')
                        assert len(frame_time) == len(frames), f'frame_time {len(frame_time)}!= frames {len(frames)}'
                        assert len(frames) > 0, f'frames {len(frames)} is empty'
                        return frame_time, frames
                except Exception as exn:
                    print(f'Exception: {exn}')

            if storage_system != 'local':
                raise NotImplementedError("Only local storage is supported for video loading.")
            vr = VideoReader(video_id_or_path, ctx=cpu(0), num_threads=8)
                    
            total_frame_num = len(vr)
            video_time = total_frame_num / vr.get_avg_fps()
            avg_fps = max(1, round(vr.get_avg_fps() / video_fps))
            if self.is_fix_frame:
                frame_idx = [i for i in range(total_frame_num)]  # Fix-Frame Sampling
            else:
                frame_idx = [i for i in range(0, total_frame_num, avg_fps)]  # FPS Sampling
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
            if frames_upbound > 0:
                if self.is_fix_frame or len(frame_idx) > frames_upbound:
                    uniform_sampled_frames = np.linspace(
                        0, total_frame_num - 1, frames_upbound, dtype=int
                    )
                    frame_idx = uniform_sampled_frames.tolist()
                    frame_time = [i / vr.get_avg_fps() for i in frame_idx]
            # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
            # print(frame_time)
            if is_accurate:
                try:
                    video = vr.get_batch(frame_idx).asnumpy()
                except Exception as exn:
                    vr = VideoReader(video_id_or_path, ctx=cpu(0), num_threads=1)
                    video = vr.get_batch(frame_idx).asnumpy()
            else:
                vr.seek(0)
                video = []
                for idx in frame_idx:
                    vr.seek(idx)
                    frame = vr.next()
                    video.append(frame.asnumpy())

            # Convert the list of frames to a numpy array if needed
            video = np.array(video)
            num_frames_to_sample = num_frames = len(frame_idx)
            n_frames, h, w, _ = video.shape
            # Convert frames to PIL Image objects and resize based on max_pixels
            pil_images = []
            for i in range(n_frames):
                # Convert numpy array to PIL Image
                img = Image.fromarray(video[i].astype('uint8'))
                
                # Calculate resize dimensions based on max_pixels
                current_pixels = h * w
                if current_pixels > max_pixels:
                    # Need to resize down
                    scale_factor = np.sqrt(max_pixels / current_pixels)
                    new_w = int(w * scale_factor)
                    new_h = int(h * scale_factor)
                    
                    # Ensure we don't go below min_pixels
                    if new_h * new_w < min_pixels:
                        min_scale_factor = np.sqrt(min_pixels / (new_h * new_w))
                        new_h = int(new_h * min_scale_factor)
                        new_w = int(new_w * min_scale_factor)
                        
                    # Resize the image
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                
                pil_images.append(img)
            print('total_frame_num:', total_frame_num, 'video_time:', video_time, 'avg_fps:', avg_fps, 'frame num:', len(pil_images), 'processing time:', time.time() - start_time)
            del vr
            assert len(frame_time) == len(pil_images), f'frame_time {len(frame_time)}!= frames {len(pil_images)}'
            assert len(pil_images) > 0, f'frames {len(pil_images)} is empty'
            return frame_time, pil_images
        except Exception as exn:
            print(f"Failed to process video {video_id_or_path}. Exception:", exn)
            return [0], [Image.open('/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/MM-Eureka/blank_image.png')] # Blank frame fallback for video load errors
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = copy.deepcopy(self.list_data_dict[item])

        chat = row_dict.pop(self.prompt_key)
        # if isinstance(chat, list) and self.system_prompt:
        #     chat.insert(0, {"role": "system", "content": self.system_prompt})
        #     chat = np.array(chat)
        # elif isinstance(chat, np.ndarray) and self.system_prompt:
        #     chat = np.insert(chat, 0, {"content": self.system_prompt, "role": "system"})

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            image_file_list = row_dict.pop(self.image_key)
            storage_system = 'local'
            # Process MultiModal Data Input
            image_grid_thw = video_grid_thw = None
            if self.image_key == 'images':
                raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                row_dict['multi_modal_data'] = {'image': [self.process_image(image_file, self.max_pixels, self.min_pixels, storage_system) for image_file in image_file_list]}
                image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
                multimodal_input_grid_thw = image_grid_thw = image_inputs['image_grid_thw']
                # row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
                place_holder_token = self.processor.image_token
            elif self.image_key == 'videos':
                assert len(image_file_list) == 1, f"Currently only support one video per sample"
                row_dict['video_files'] = image_file_list
                if self.pass_video_as_frames:
                    image_file = image_file_list[0]
                    if self.random_fps:
                        keys = list(self.random_fps.keys())
                        values = list(self.random_fps.values())
                        frame_num = int(random.choices(keys, weights=values, k=1)[0])
                        max_pixels = min(int(self.max_pixels * self.frames_upbound // frame_num), 100352)
                        min_pixels = 25088 if max_pixels > 25088 else 12544
                        frame_time, frames = self.process_video(image_file, self.video_fps, frame_num, max_pixels, min_pixels, storage_system, is_accurate=(not self.fast_seek))
                    else:
                        frame_time, frames = self.process_video(image_file, self.video_fps, self.frames_upbound, self.max_pixels, self.min_pixels, storage_system, is_accurate=(not self.fast_seek))
                    raw_prompt = prompt_with_chat_template.replace('<image>', ''.join([f'<frame{i}_time{frame_time[i]:.2f}s><|vision_start|><|image_pad|><|vision_end|>' for i in range(len(frames))]))
                    prompt_with_chat_template = prompt_with_chat_template.replace('<image>', ''.join([f'<frame{i}_time{frame_time[i]:.2f}s><image>' for i in range(len(frames))]))
                    row_dict['multi_modal_data'] = {'image': frames}
                    row_dict['prompt_img_num'] = len(frames)
                    image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
                    multimodal_input_grid_thw = image_grid_thw = image_inputs['image_grid_thw']
                    # row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
                    place_holder_token = self.processor.image_token
                else:
                    raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|video_pad|><|vision_end|>')
                    row_dict['multi_modal_data'] = {'video': [self.process_video(image_file, self.video_fps, self.frames_upbound, self.max_pixels, self.min_pixels, storage_system, is_accurate=(not self.fast_seek))[1] for image_file in image_file_list]}
                    row_dict['prompt_img_num'] = 0
                    row_dict['multi_modal_data']['image'] = []
                    video_inputs = self.processor.image_processor(images=None, videos=row_dict['multi_modal_data']['video'], return_tensors='pt')
                    multimodal_input_grid_thw = video_grid_thw = video_inputs['video_grid_thw']
                    # row_dict['multi_modal_inputs'] = {key: val for key, val in video_inputs.items()}
                    place_holder_token = self.processor.video_token
            else:
                raise ValueError(f"Unknown image_key: {self.image_key}")

            if multimodal_input_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (multimodal_input_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>', place_holder_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal and self.is_qwen:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        ### Add for CustomRewardManager ###
        row_dict['ground_truth'] = row_dict.pop('solution')

        # encode prompts without chat template
        if self.return_raw_chat:
            assert chat[-1]['role'] == 'user'
            row_dict['raw_prompt'] = chat[-1]['content']


        # add index for each prompt
        index = row_dict.get("doc_id", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'list_data_dict' in state:
                del state['list_data_dict']
            return state
        return self.__dict__.copy()


if __name__ == '__main__':
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor
    from multiprocessing import Pool
    from loguru import logger
    import uuid
    local_path = copy_to_local("Qwen/Qwen2.5-VL-3B-Instruct")
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none
    # SYSTEM_PROMPT="You FIRST think about the reasoning process step by step as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in <answer> </answer> tags."
    SYSTEM_PROMPT=None
    # IMAGE
    # dataset = MultiModalDataset(
    #     # data_path="/mnt/bn/tiktok-mm-5/aiic/users/lijunyi/data/MultiModalGRPOData/yaml/mm_rl_train_single_image.yaml",
    #     data_path="/mnt/bn/tiktok-mm-5/aiic/users/lijunyi/data/MultiModalGRPOData/yaml/MMEureka_MMR1_train.yaml",
    #     tokenizer=tokenizer,
    #     processor=processor,
    #     prompt_key='prompt',
    #     image_key='images',
    #     max_prompt_length=4096,
    #     filter_prompts=True,
    #     return_raw_chat=False,
    #     truncation='error',
    #     system_prompt=SYSTEM_PROMPT,
    #     max_pixels=2048 * 2048,
    #     min_pixels=512 * 512,
    # )
    # VIDEO
    # ray.init()
    # dataset = MultiModalDataset(
    #     data_path="/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/MLVU/train.yaml",
    #     tokenizer=tokenizer,
    #     processor=processor,
    #     prompt_key='prompt',
    #     image_key='videos',
    #     max_prompt_length=131072,
    #     filter_prompts=True,
    #     return_raw_chat=False,
    #     truncation='error',
    #     system_prompt=SYSTEM_PROMPT,
    #     max_pixels=100352,
    #     min_pixels=512,
    #     video_fps=0.2,
    #     is_fix_frame=False,
    #     frames_upbound=30,
    # )
    # from pyinstrument import Profiler
    # profiler = Profiler()
    # profiler.start()
    # frametime, video = dataset.process_video('/mnt/bn/tiktok-mm-5/aiic/users/dingyang/data/MVLU/MLVU/video/1_plotQA/movie101_66.mp4', is_accurate=True)
    # # dar 
    # # frametime, video2 = dataset.process_video('/opt/tiger/aiic_verl/temp.mp4')
    # # save as gif

    # # print(len(video))
    # # import time
    # # start = time.time()
    # # for i in range(10):
    # #     example = dataset[i]
    # # end = time.time()
    # # print(f'time: {end-start}') # 视频读取速度测试

    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
    # print(frametime)
    # print(len(dataset[0]['multi_modal_data']))
    # print(processor.image_processor(dataset[0]['multi_modal_data']['image']))
    train_dataset = MultiModalDataset(
        data_path='/mnt/bn/tiktok-mm-5/aiic/users/dingyang/data/longvideo-reason/train_processed_filtered.json',
        tokenizer=tokenizer,
        processor=processor,
        prompt_key='prompt',
        image_key='videos',
        max_prompt_length=131072,
        filter_prompts=True,
        return_raw_chat=False,
        truncation='error',
        system_prompt=SYSTEM_PROMPT,
        post_prompt=None,
        is_qwen=True,
        max_pixels=100352,
        min_pixels=25088,
        pass_video_as_frames=True,
        video_fps=0.5,
        frames_upbound=64,
    )
    cache_path = f'/mnt/bn/tiktok-mm-5/aiic/users/dingyang/data/preprocessed_frames/fps{train_dataset.video_fps}_upbound{train_dataset.frames_upbound}_maxpixels{train_dataset.max_pixels}_minpixels{train_dataset.min_pixels}'
    data = json.load(open('/mnt/bn/tiktok-mm-5/aiic/users/dingyang/data/longvideo-reason/test_processed.json'))
    # Define a helper function for parallel processing
    def process_video_wrapper(item):
        video_path = item['videos'][0]
        save_path = f"{cache_path}/{uuid.uuid5(uuid.NAMESPACE_DNS, video_path)}"
        if os.path.exists(f"{save_path}/frame_time.txt"):
            logger.info(f"{video_path} already processed and saved to cache {save_path}")
            return 0
        frame_time, frames = train_dataset.process_video(
            video_path, 
            video_fps=train_dataset.video_fps,
            frames_upbound=train_dataset.frames_upbound,
            max_pixels=train_dataset.max_pixels, 
            min_pixels=train_dataset.min_pixels, 
            storage_system='local',
            preprocess=True,
        )
        # Save the frames to the cache
        os.makedirs(save_path, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(f"{save_path}/frame{i}.png")
        np.savetxt(f"{save_path}/frame_time.txt", frame_time)
        logger.success(f"{video_path} processed and saved to cache {save_path}")
        return len(frames)
    import random
    random.shuffle(data)
    # Create a pool of workers (adjust the number based on your CPU cores)
    with Pool(processes=16) as pool:
        # Process the first 10 items in parallel
        results = list(tqdm(pool.imap(process_video_wrapper, data), total=len(data)))
    
    # # Print the results
    # print(results)
    
    # process_video_wrapper(data[0])

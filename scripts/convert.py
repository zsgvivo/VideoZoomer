import json
import glob
import os
import torch

import argparse

from transformers import (
    Qwen2VLConfig, Qwen2VLForConditionalGeneration, AutoTokenizer,
    AddedToken, Qwen2VLProcessor, Qwen2VLImageProcessor,
)

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from accelerate import init_empty_weights

chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|image_pad|>\n{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|video_pad|>\n{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

def load_original_state_dict(model_id):
    original_state_dict = {}
    for path in glob.glob(f"{model_id}/*"):
        if path.endswith(".bin"):
            f = torch.load(path)
            for key in f.keys():
                original_state_dict[key] = f.get(key)
    # tied wieghts so lm.head is not saved. Let's clone to load state dict
    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()

    return original_state_dict

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.vision_tower": "visual",
}

def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        for old_key, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if old_key in key:
                key = key.replace(old_key, new_key)
        new_state_dict[key] = value
    return new_state_dict


def convert_llava_qwenvl(model_id, pytorch_dump_path):

    
    config_path = os.path.join(model_id, "config.json")
    with open(config_path) as f:
        data = json.load(f)

    vision_config = Qwen2VLVisionConfig(
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        in_chans = 3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        spatial_patch_size=14,
    ).to_dict()

    config = Qwen2VLConfig(
        hidden_size=data["hidden_size"],
        vocab_size=data["vocab_size"],
        intermediate_size=data["intermediate_size"],
        num_hidden_layers=data["num_hidden_layers"],
        num_attention_heads=data["num_attention_heads"],
        num_key_value_heads=data["num_key_value_heads"],
        hidden_act=data["hidden_act"],
        max_position_embeddings= 131072, #data["max_position_embeddings"],
        attention_dropout=data['attention_dropout'],
        initializer_range=data['initializer_range'],
        max_window_layers=data['max_window_layers'],
        rope_scaling=data['rope_scaling'],
        bos_token_id = 151643,
        eos_token_id = 151645,
        vision_start_token_id = 151652,
        vision_end_token_id = 151653,
        vision_token_id = 151654,
        image_token_id = 151655,
        video_token_id = 151656,
        chat_modify  = True,

        # rope_scaling = {
        #     "mrope_section": [16,24,24],
        #     "rope_type": "default",
        #     "type": "default"
        # },

        vision_config=vision_config,
    )
    text_model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=False)

    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_tokens(AddedToken("<video>", special=True, normalized=False), special_tokens=True)

    tokenizer.chat_template = chat_template

    vision_model_id = data["mm_vision_tower"]
    min_pixels = data["min_pixels"]
    max_pixels = data["max_pixels"]

    image_processor = Qwen2VLImageProcessor(
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    # image_processor.__class__.__name__ = "Qwen2VLImageProcessorGPU"
    image_processor.__class__.__name__ = "Qwen2VLImageProcessorFastCPU"

    processor = Qwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    with init_empty_weights():
        model = Qwen2VLForConditionalGeneration(config)
        model.__class__.__name__ = "TTVFMForConditionalGeneration"

    original_state_dict = load_original_state_dict(model_id)
    new_state_dict = convert_state_dict_to_hf(original_state_dict)
    model.load_state_dict(new_state_dict, assign=True)

    model.save_pretrained(pytorch_dump_path)
    processor.save_pretrained(pytorch_dump_path)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_id", type=str, default="/opt/tiger/maas_engine/workspace/model2")
    parser.add_argument("--pytorch_dump_path", type=str, default="/opt/tiger/maas_engine/workspace/model_convert3")

    args = parser.parse_args()
    convert_llava_qwenvl(model_id = args.model_id, pytorch_dump_path=args.pytorch_dump_path)

if __name__ == "__main__":
    main()
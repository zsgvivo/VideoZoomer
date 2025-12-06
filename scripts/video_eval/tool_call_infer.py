from function_tools import extract_video_clip, prepare_tool_call_inputs_video
import openai
import subprocess
import os
import requests
import base64
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging
import asyncio
import pandas as pd
from decord import VideoReader, cpu
from io import BytesIO
from loguru import logger
import cv2
from PIL import Image
import numpy as np    
from tqdm.asyncio import tqdm_asyncio
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
from collections import defaultdict
import argparse
import json
from multi_client import MultiClientManager


def _require_env(var_name: str) -> str:
    """Fetch an environment variable or raise if missing."""
    value = os.environ.get(var_name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return value


azure_openai_endpoint = _require_env("AZURE_OPENAI_VISION_ENDPOINT")
azure_openai_key = _require_env("AZURE_OPENAI_VISION_API_KEY")
azure_openai_version = os.environ.get("AZURE_OPENAI_VISION_API_VERSION", "2024-09-01-preview")
gptclient = openai.AsyncAzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_version=azure_openai_version,
    api_key=azure_openai_key,
)
gptmodel = os.environ.get("AZURE_OPENAI_VISION_MODEL", "gpt-4o-2024-11-20")

gemini_endpoint = os.environ.get("AZURE_GEMINI_ENDPOINT")
gemini_api_key = os.environ.get("AZURE_GEMINI_API_KEY")
gemini_api_version = os.environ.get("AZURE_GEMINI_API_VERSION", "2024-03-01-preview")
geminimodel = os.environ.get("AZURE_GEMINI_MODEL", "gemini-2.5-pro-preview-06-05")
geminiclient = (
    openai.AsyncAzureOpenAI(
        azure_endpoint=gemini_endpoint,
        api_version=gemini_api_version,
        api_key=gemini_api_key,
    )
    if gemini_endpoint and gemini_api_key
    else None
)


from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def read_img_as_base64(img):
    if isinstance(img, str):
        pil_img = Image.open(img)
    else:
        pil_img = img
    buffered = BytesIO()
    format = "PNG" if pil_img.mode in ("RGBA", "LA") else "JPEG"
    pil_img.save(buffered, format=format)
    return f"data:image/{format.lower()};base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def judge_multi_choice(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
    if ground_truth[-1] == '.':
        ground_truth = ground_truth[:-1]
    extracted_str = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    if extracted_str is not None:
        predict_str = extracted_str
    def extract_choice(answer):
        import re
        pattern = r"([A-F])\."
        matches = re.findall(pattern, answer, re.MULTILINE)
        if matches:
            return matches[-1]  # 返回最后一个匹配项
        else:
            return None
    accuracy_reward = 0.0
    if predict_str == ground_truth:
        accuracy_reward = 1.0
    if accuracy_reward == 0.0:
        try:
            extracted_answer = extract_boxed_content(predict_str)
            if extracted_answer == ground_truth:
                accuracy_reward = 1.0
        except Exception as e:
            pass
    if accuracy_reward == 0.0:
        try:
            extracted_answer = extract_choice(predict_str)
            if extracted_answer == ground_truth:
                accuracy_reward = 1.0
        except Exception as e:
            pass
    accuracy_reward_dict = {
        "accuracy": accuracy_reward,
    }
    return accuracy_reward_dict

SYSTEM_PROMPT = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\nYour task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n- Focus on the meaningful match between the predicted answer and the correct answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the correctness of the prediction compared to the answer."

QUERY_PROMPT = """I will give you a question related to an image and the following text as inputs:\n\n1. **Question Related to the Image**: {question}\n2. **Ground Truth Answer**: {ground_truth}\n3. **Model Predicted Answer**: {prediction}\n\nYour task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?\n- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:\n(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.\n(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct.\n**Output Format**:\nYour response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model's prediction strictly aligns with the ground truth, while 0 means it does not.\nThe format should be \"Score: 0 or 1\""""
async def judge_open_ended(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
    predict_str = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    accuracy_reward = 0.0
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": QUERY_PROMPT.format(question=prompt_str, ground_truth=ground_truth, prediction=predict_str)
        }
    ]
    response = await query_client_with_retry(gptclient, gptmodel, messages)
    response = response[0]
    logger.debug(f"gpt response: {response}")
    if 'score:' not in response.lower():
        # raise ValueError(f"No 'score:' in response: {response}")
        score = 0
        # print("Extraction error, response: ", response)
        logger.warning(f"Extraction error, response: {response}")
    extract_score = response.lower().split("score:")[-1].strip().split("\n")[0].strip().split(" ")[0]
    if "1" not in extract_score and '0' not in extract_score:
        score = 0
        # print("Extraction error, response: ", response)
        logger.warning(f"Extraction error, response: {response}")
    else:
        score = 1.0 if '1' in extract_score else 0.0

    accuracy_reward_dict = {
        "accuracy": score,
    }
    return accuracy_reward_dict
def process_video(video_id_or_path: str, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512, storage_system: str = 'local', video_fps=0.2, frames_upbound=30):
    from io import BytesIO
    import cv2
    from PIL import Image
    import numpy as np
    
    # Currently, we do not support load precomputed features
    if storage_system != 'local':
        raise NotImplementedError("Only local video loading is supported.")
    vr = VideoReader(video_id_or_path, ctx=cpu(0), num_threads=8)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]  # FPS Sampling
    frame_time = [i / vr.get_avg_fps() for i in frame_idx]

    if frames_upbound > 0:
        if len(frame_idx) > frames_upbound:
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, frames_upbound, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    video = vr.get_batch(frame_idx).asnumpy()
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
    logger.debug(f"total_frame_num: {total_frame_num}, video_time: {video_time}, avg_fps: {avg_fps}, frame num: {len(pil_images)}")
    return pil_images

async def process_video_async(video_id_or_path: str, max_pixels: int = 2048 * 2048, min_pixels: int = 25088, storage_system: str = 'local', video_fps=0.2, frames_upbound=30):
    # 异步执行IO密集型操作
    def sync_video_processing():
        try:
            # 这部分是同步操作，但将在线程池中运行
            if storage_system != 'local':
                raise NotImplementedError("Only local video loading is supported.")
            vr = VideoReader(video_id_or_path, ctx=cpu(0), num_threads=1)
            
            total_frame_num = len(vr)
            video_time = total_frame_num / vr.get_avg_fps()
            avg_fps = round(vr.get_avg_fps() / video_fps)
            frame_idx = [i for i in range(0, total_frame_num, avg_fps)]  # FPS Sampling
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]

            if frames_upbound > 0:
                if len(frame_idx) > frames_upbound:
                    uniform_sampled_frames = np.linspace(
                        0, total_frame_num - 1, frames_upbound, dtype=int
                    )
                    frame_idx = uniform_sampled_frames.tolist()
                    frame_time = [i / vr.get_avg_fps() for i in frame_idx]

            video = vr.get_batch(frame_idx).asnumpy()
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
            
            logger.debug(f"total_frame_num: {total_frame_num}, video_time: {video_time}, avg_fps: {avg_fps}, frame num: {len(pil_images)}")
            return pil_images, frame_time
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return [], []
    
    # 在事件循环的默认线程池中执行IO密集型操作
    return await asyncio.to_thread(sync_video_processing)

logger_std = logging.getLogger(__name__)
@retry(
    stop=stop_after_attempt(10),  # 最大重试次数
    wait=wait_exponential(multiplier=1, min=1, max=64),  # 指数退避 (1, 2, 4, 8, ..., 128)
    retry=(retry_if_exception_type(Exception)),  # 对所有异常重试
    before_sleep=before_sleep_log(logger_std, logging.WARNING),  # 重试前记录日志
    reraise=True  # 重试结束后重新抛出异常
)
async def query_client_with_retry(client, model, messages, semaphore=None, video_kwargs=None, rollout_num: int = 1,timeout=60):
    chat_response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=1.0,
                    max_tokens=1024,
                    timeout=timeout,
                    n=rollout_num,
                    extra_body={
                        "mm_processor_kwargs": video_kwargs
                    }
                )
    # response_str = chat_response.choices[0].message.content
    response_list = [chat_response.choices[i].message.content for i in range(rollout_num)]
    return response_list
import copy
import re
@retry(
    stop=stop_after_attempt(10),  # 最大重试次数
    wait=wait_exponential(multiplier=1, min=1, max=64),  # 指数退避 (1, 2, 4, 8, ..., 128)
    retry=(retry_if_exception_type(Exception)),  # 对所有异常重试
    before_sleep=before_sleep_log(logger_std, logging.WARNING),  # 重试前记录日志
    reraise=True  # 重试结束后重新抛出异常
)
async def multiturn_query_client_with_tool(client, model, messages,video_path , semaphore=None, video_kwargs=None,timeout=60,max_pixels=100352,min_pixels=25088,tool_call_max_frames=16, max_generation_round=2):
    messages = copy.deepcopy(messages)
    result = None
    round = 0
    while round < max_generation_round:
        round += 1
        chat_response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=1.0,
                        max_tokens=8096,
                        timeout=timeout,
                        # logprobs=True,
                        # stop='</video_zoom>',
                    )
        # chat_response = await client_manager.chat(
        #                 # model=model,
        #                 messages=messages,
        #                 temperature=1.0,
        #                 max_tokens=8096,
        #                 timeout=timeout,
        #                 # logprobs=True,
        #                 # stop='</video_zoom>',
        #             )
        res = chat_response.choices[0].message.content
        result = res.split('</video_zoom>')[0]
        if len(res.split('</video_zoom>')) > 1:
            result += '</video_zoom>'
            if len(res.split('</video_zoom>')[1]) > 0:
                logger.warning(f"response did not stop after </video_zoom>, {res.split('</video_zoom>')[-1]}, len(res.split('</video_zoom>'))={len(res.split('</video_zoom>'))}")
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": result
                }
            ]
        })
        # if chat_response.choices[0].message.content.endswith('</video_zoom>'): #FIXME openai does not support include_stop_str (https://github.com/InternLM/lmdeploy/issues/1731)
        pattern = re.compile(r'<video_zoom>(.*?)</video_zoom>', re.DOTALL)
        tool_call_contents = pattern.findall(result)
        if len(tool_call_contents) > 0:
            logger.debug(f"Trying to call function tools...")
            error_info = None
            try:
                json_pattern = re.compile(r'\{.*?\}')
                json_objects = []
                for content in tool_call_contents:
                    json_strings = json_pattern.findall(content)
                    json_objects.extend([json.loads(json_str) for json_str in json_strings])
                tool_info = prepare_tool_call_inputs_video(json_objects)
            except Exception as e:
                logger.warning(str(e))
                error_info = str(e)
                tool_info = None
            if error_info is not None:
                tool_outputs = f"\nERROR occurs during the function tool call. Error Information: {error_info}.\n"
            else:
                logger.debug(f"Calling function tools..., {tool_info}")
                tool_outputs = extract_video_clip(video_path=video_path, **tool_info, storage_system='local', max_pixels=max_pixels, min_pixels=min_pixels, max_frames=tool_call_max_frames,logger=logger)
            if not isinstance(tool_outputs, str):
                frame_time = tool_outputs['frame_time']
                frames = tool_outputs['frames']
                messages.append({
                    "role": "user",
                    "content": [{f"type":"text", "text": "The frames of the video clip are shown below:\n"}]
                    }
                )
                for i in range(len(frames)):
                    messages[-1]['content'].append({
                        "type": "text",
                        "text": f"<frame{i}_time{frame_time[i]:.2f}s>"
                    })
                    messages[-1]['content'].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": read_img_as_base64(frames[i])
                                }
                            })
                messages[-1]['content'].append({
                    "type": "text",
                    "text": "continue your reasoning process inside <think> and </think> and then write your final answer inside <answer> and </answer>.",
                    }
                )
            else:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": tool_outputs + "Please analyze the error information obtained from the function tool and adjust your response. Countinue your reasoning process inside <think> and </think> and then write your final answer inside <answer> and </answer>."
                        },
                    ]
                })
            if round == max_generation_round-1:
                messages[-1]['content'].append({
                    "type": "text",
                    "text": "Do not call <video_zoom> in this round, give a final answer based on information above."
                })
        else:
            break
        # logger.debug(extract_text_from_messages(messages))
    return result, messages, round
    # response_list = [chat_response.choices[i].message.content for i in range(rollout_num)]
    # return response_list


def extract_text_from_messages(messages):
    text = []
    for message in messages:
        if message["role"] == "user" or message["role"] == "assistant":
            content = message["content"]
            if isinstance(content, str):
                text.append(f"{message['role'].upper()}: {content}")
            elif isinstance(content, list):
                text.append(f"{message['role'].upper()}: ")
                for item in content:
                    if isinstance(item, dict):
                        text[-1] += item.get("text", "")
                    elif isinstance(item, str):
                        text[-1] += item
    return text
import mimetypes
def extract_message_to_sft_format(messages, image_dir, prompt_image_num = 0, prompt_image_paths = []):
    assert prompt_image_num == len(prompt_image_paths), f"prompt_image_num {prompt_image_num}!= len(prompt_image_paths) {len(prompt_image_paths)}"
    os.makedirs(image_dir, exist_ok=True)
    img_idx = 0
    image_paths = []
    sft_messages = []
    for message in messages:
        role = message["role"]
        content_parts = message["content"]
        full_content = ""
        # if not isinstance(content_parts, list):
        #     import pdb; pdb.set_trace()
        if isinstance(content_parts, str):
            full_content += content_parts
            sft_messages.append({"role": role, "content": full_content})
            continue
        assert isinstance(content_parts, list), f"content_parts {content_parts} is not a list"
        for part in content_parts:
            
            if part["type"] == "text":
                full_content += part["text"]
            elif part["type"] == "image_url":
                if img_idx < prompt_image_num:
                    image_path = prompt_image_paths[img_idx]
                    img_idx += 1
                    image_paths.append(image_path)
                    full_content += "<image>"
                    continue
                # 解析图像 URL
                image_url = part["image_url"]['url']
                try:
                    match = re.match(r'data:(image/.+?);base64,(.+)', image_url)
                    mime_type = match.group(1)
                    base64_data = match.group(2)

                    
                    # 保存图片
                    # 生成唯一文件名
                    image_data = base64.b64decode(base64_data)
                    image_path = os.path.join(image_dir, f"{img_idx}.png")  # 强制使用.png扩展名
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    # logger.debug(f"Image {img_idx}.png saved successfully to {image_path}.")
                    # 添加相对路径到列表
                    image_paths.append(os.path.join(image_dir, f"{img_idx}.png"))
                    img_idx += 1

                    full_content += "<image>"
                except Exception as e:
                    # 若解码失败，跳过此图片
                    logger.error(f"Error decoding image {img_idx}.png: {str(e)}")
        sft_messages.append({"role": role, "content": full_content})
    # remove few shots
    sft_messages[1]['content'] = sft_messages[1]['content'].split('Now answer the following question:')[-1]
    image_paths = image_paths[:]
    return sft_messages, image_paths


from qwen_vl_utils import process_vision_info

async def encode_frame_async(frame: np.ndarray) -> str:
    """异步编码单帧图像为 base64"""
    def _encode():
        img = Image.fromarray(frame)
        buffer = BytesIO()
        img.save(buffer, format="jpeg")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return await asyncio.to_thread(_encode)

async def prepare_message_for_vllm_async(content_messages):
    vllm_messages, fps_list = [], []

    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # 异步编码所有帧
                base64_frames = await asyncio.gather(*[encode_frame_async(frame) for frame in video_input])

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}
import uuid
def load_example(path, include_syspmt=False):
    example = json.load(open(path))
    messages = example[0]['messages']
    prompt_images = example[0]['images']
    if not include_syspmt:
        messages = messages[1:]
    res = []
    img_idx = 0
    for message in messages:
        res.append({
                    "role": message['role'],
                    "content": []
                })
        text_splits = message['content'].split('<image>')
        for i, text_split in enumerate(text_splits):
            res[-1]['content'].append({"type": "text", "text": text_split})
            if i < len(text_splits) - 1:
                # print(img_idx)
                img_data = read_img_as_base64(prompt_images[img_idx])
                res[-1]['content'].append({"type": "image_url", "image_url": {"url": img_data}})
                img_idx += 1
    assert img_idx == len(prompt_images)
    return res, prompt_images
async def evaluate(video_path, problem, answer, extra_info, semaphore, client, model, rollout_num: int = 1, fps=0.2, max_pixels=100352, pass_video_as_frames: bool = True, frames_upbound=30, timeout=60, use_fewshots=True, max_generation_round=2, save_sft=True, tool_call_max_frames=16):
    async with semaphore:  # 自动获取和释放信号量
        detailed_res = []
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. You will receive a low-frame-rate video and related questions. You can analyze the video content to answer the question and trigger high-frame-rate inspections when finer temporal resolution is needed. When you detect ambiguous motion/objects that require closer inspection, wrap your request in <video_zoom></video_zoom> tags and provide the exact time segment and target frame rate in JSON format: <video_zoom> {\"segment\": [start_sec, end_sec], \"fps\": n} </video_zoom>, it will return the video clip at the target fps to help you better answer the question. Note that the total frames num of the request clip cannot exceed {} (e.g., (end_sec - start_sec) * fps ≤ 16) and DO NOT include <answer> tags in this round. \n Example usage: <video_zoom> {\"segment\": [4.0, 6.0], \"fps\": 2} </video_zoom>.\n\nIf the initial tool response doesn't provide sufficient information to answer the question, you may continue to request additional video zoom inspections as needed, until you either (1) gather enough information to form a complete answer, or (2) are explicitly instructed to stop using the tool.\n Output the thinking process within <think> </think> tags, once you confirm your final answer, place the final answer in \\boxed{{}} inside <answer> and </answer>.",
                    }
                ]
            },
            {
                "role": "user",
                "content": [],
            }
        ]
        if pass_video_as_frames:

            if use_fewshots:
                prompt_images = []
                messages[-1]['content'].append({
                    "type": "text",
                    "text": "Here are some examples:\n"
                })
                idx = 0
                for fewshot_idx in [6,7]:
                    messages[-1]['content'].append({
                        "type": "text",
                        "text": f"Example {idx}: \n",
                    })
                    idx += 1
                    # ex_msg = json.load(open(f"/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/ytb_json_files/sft/examples/{fewshot_idx}.json"))
                    ex_msg, imgs = load_example(f"/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/ytb_json_files/sft/examples/{fewshot_idx}.json")
                    prompt_images.extend(imgs)
                    for msg in ex_msg:
                        if isinstance(msg['content'], str):
                            messages[-1]['content'].append({
                                "type": "text",
                                "text": msg['content']
                            })
                        elif isinstance(msg['content'], list):
                            messages[-1]['content'].extend(msg['content'])
                    messages[-1]['content'].append({
                        "type": "text",
                        "text": "\n"
                    })
                #     messages.extend(ex_msg)
                # messages.append({
                #     "role": "user",
                #     "content": []
                # })
          
                messages[-1]['content'].append({
                    "type": "text",
                    "text": "—--"* 10 + "\n Now answer the following question: \n"
                })

            if fps > 0:
                video_frames, frame_time = await process_video_async(video_path, max_pixels=max_pixels, video_fps=fps, frames_upbound=frames_upbound)
                for idx, frame in enumerate(video_frames):
                    messages[-1]['content'].append({
                        "type": "text",
                        "text": f"<frame{idx}_time{frame_time[idx]:.2f}s>"
                    })
                    messages[-1]['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": read_img_as_base64(frame),
                        }
                    }) 
            messages[-1]['content'].append({
                        "type": "text",
                        "text": problem, 
                    })
            video_kwargs = None
            for item in messages[-1]['content']:
                if "text" in item:
                    item["text"] = item["text"].replace("<image>", "")
        else:
            # raise NotImplementedError
            messages[-1]['content'] = [
                {
                    "type": "video",
                    "video" :"file://" + video_path,
                    "max_pixels": max_pixels,
                    "fps": fps,
                    "max_frames": frames_upbound
                },
                {"type": "text", "text": problem}]
            # messages, video_kwargs = prepare_message_for_vllm(messages)
            messages, video_kwargs = await prepare_message_for_vllm_async(messages)

        for i in range(rollout_num):
            try:
                pred, new_messages, round_n = await multiturn_query_client_with_tool(client, model, messages,video_path, video_kwargs,timeout=timeout, max_pixels=max_pixels, max_generation_round=max_generation_round, tool_call_max_frames=tool_call_max_frames)
                # import pdb; pdb.set_trace()
                res = {}
                if extra_info['answer_type'] == 'multiple_choice':
                    res = judge_multi_choice(problem, pred, answer)
                elif extra_info['answer_type'] =='open_ended':
                    res = await judge_open_ended(problem, pred, answer)
                else:
                    raise NotImplementedError
                res['pred'] = pred
                res['messages'] = extract_text_from_messages(new_messages)
                import time
                if save_sft:
                    unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_path}_{problem}_{answer}_{i}")
                    sft_res = extract_message_to_sft_format(new_messages, f"/mnt/bn/tiktok-mm-5/aiic/users/dingyang/data/sft_images/{now}/{unique_id}", prompt_image_num=(len(prompt_images) if use_fewshots else 0), prompt_image_paths=prompt_images)
                    res['sft_messages'] = {
                        "messages": sft_res[0],
                        "images": sft_res[1]
                    }
                res['round'] = round_n
                res['rollout_index'] = i
                detailed_res.append(res)
            except Exception as e:
                logger.error(f"Error processing video {video_path} with problem '{problem}': {str(e)}")
        pass_rate = sum([r['accuracy'] for r in detailed_res]) / (len(detailed_res) + 1e-5)
        logger.debug(f'pass_rate: {pass_rate}')
        return {'pass_rate': pass_rate, 'detailed_rollout': detailed_res}


async def main():
    # json_path = '/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/MathVision/oe_train_subproblem_gpt4o_verified.json'
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Video Evaluation Script')
    
    # 添加命令行参数
    parser.add_argument('--json_path', type=str, default='/opt/tiger/aiic_verl/scripts/video_eval/test.json',
                        help='Path to the JSON file containing the dataset')
    parser.add_argument('--fps', type=float, default=0.2,
                        help='Frames per second for video processing')
    parser.add_argument('--max_pixels', type=int, default=100352,
                        help='Maximum number of pixels for resizing video frames')
    parser.add_argument('--eval_rollout_num', type=int, default=1,
                        help='Number of evaluation rollouts per video')
    parser.add_argument('--model', type=str, default='gpt-4o-2024-11-20',
                        help='model name to eval')
    parser.add_argument('--openai_api_base', type=str, default='http://localhost:8000/v1',
                        help='openai_api_base')
    parser.add_argument('--max_concurrency', type=int, default=20,
                        help='Maximum number of concurrent tasks')
    parser.add_argument('--save_results', action='store_true', 
                    help='Save results to a JSON file (default: False)')
    parser.add_argument('--pass_video_as_frames', action='store_true',
                    help='Pass video as frames to vLLM (default: False)')
    parser.add_argument('--frames_upbound', type=int, default=120,
                    help='Frames upbound for video processing')
    parser.add_argument('--request_timeout', type=int, default=120,
                    help='Request timeout for video processing')
    parser.add_argument('--max_generation_round', type=int, default=2,
                    help='Maximum number of generation rounds for tool calling')
    parser.add_argument('--save_sft', action='store_true',
                    help='Save sft messages to a JSON file (default: False)')
    parser.add_argument('--tool_call_max_frames', type=int, default=16,
                    help='Maximum number of frames for tool calling')
    parser.add_argument('--batch_size', type=int, default=-1,
                    help='batch size for data processing')
    parser.add_argument('--no_fewshots', action='store_true',  # 默认不使用时为 True
                    help='Disable fewshots for training')
    parser.add_argument('--name', type=str, default='test')
    args = parser.parse_args()
    if args.model == gptmodel:
        client = gptclient
        logger.info("using gpt model")
    elif args.model == geminimodel:
        if geminiclient is None:
            raise EnvironmentError("Gemini client is not configured. Set AZURE_GEMINI_* environment variables.")
        client = geminiclient
        logger.info("using gemini model")
    else:
        openai_api_key = "EMPTY"
        client = openai.AsyncOpenAI(
            api_key=openai_api_key,
            base_url=args.openai_api_base,
        )
    df = pd.read_json(args.json_path)
    df.head()
    train_set = df.to_dict(orient='records')[:]
    semaphore = asyncio.Semaphore(args.max_concurrency)   # 限制并发数

    counter = defaultdict(int)
    round_counter = defaultdict(int)
    batch_size = args.batch_size if args.batch_size > 0 else len(train_set)
    num_batches = (len(train_set) + batch_size - 1) // batch_size
    base_name = os.path.basename(args.json_path).replace('.json', '')
    model_name = os.path.basename(args.model).replace('/', '_')
    save_dir = os.path.join(os.path.dirname(args.json_path), 'eval', f"toolcall_{base_name}_model{model_name}_eval_rollout{args.eval_rollout_num}_fps{args.fps}_max_pixels{args.max_pixels}_max_round{args.max_generation_round}_{args.name}")
    # save model name
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'model_name.txt'), 'w') as f:
        f.write(args.model)


    all_results = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(train_set))
        batch_data = train_set[start:end]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({start}-{end})")
        base_name = os.path.basename(args.json_path).replace('.json', '')
        model_name = os.path.basename(args.model).replace('/', '_')
        batch_save_path = os.path.join(
            save_dir,
            f'batch_{batch_idx + 1}_of_{num_batches}.json'
        )
        if os.path.exists(batch_save_path):
            logger.info(f"Batch {batch_idx + 1} already processed. Skipping.")
            # 把旧结果读回来，用于最后 merge
            with open(batch_save_path, 'r') as f:
                batch_results = json.load(f)
            all_results.extend(batch_results)
            for item in batch_results:
                counter[item['pass_rate']] += 1
                for rollout in item.get('detailed_rollout', []):
                    round_counter[rollout['round']] += 1
            continue

        tasks = []
        for item in batch_data:
            video_url = item['videos'][0]
            task = asyncio.create_task(evaluate(
                video_url, item['problem'], str(item['solution']), item['extra_info'], semaphore,
                client, args.model, args.eval_rollout_num, args.fps, args.max_pixels,
                pass_video_as_frames=args.pass_video_as_frames,
                frames_upbound=args.frames_upbound,
                timeout=args.request_timeout,
                max_generation_round=args.max_generation_round,
                save_sft=args.save_sft,
                tool_call_max_frames=args.tool_call_max_frames,
                use_fewshots=not args.no_fewshots,
            ))
            tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks)

        for idx, result in enumerate(results):
            batch_data[idx]['pass_rate'] = result['pass_rate']
            counter[result['pass_rate']] += 1
            batch_data[idx]['detailed_rollout'] = result['detailed_rollout']
            for item in result['detailed_rollout']:
                round_counter[item['round']] += 1
            all_results.append(batch_data[idx])

        # ✅ 保存一次中间结果
        if args.save_results and num_batches > 1:
            with open(batch_save_path, 'w') as f:
                json.dump(batch_data, f, indent=4)
            logger.info(f"Batch {batch_idx + 1} saved to {batch_save_path}")

    logger.info(len(results))
    logger.info(f"pass rate: {counter}")
    logger.info(f"round_number: {round_counter}")

    if args.save_results:
        # ✅ 合并所有 batch 结果为最终输出
        final_save_path = os.path.join(
            save_dir,
            f'merged_toolcall_result.json'
        )
        with open(final_save_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Final merged results saved to {final_save_path}")
        # save pass rate
        save_dir2 = os.path.join(save_dir, f'toolcall_passrate.json')
        with open(save_dir2, 'w') as f:
            json.dump(counter, f, indent=4)
        logger.info(f'pass rate saved to {save_dir2}')
    return 

if __name__ == "__main__":
    asyncio.run(main())

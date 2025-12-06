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
    """Fetch an environment variable or raise if the credential is missing."""
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

REFLECTION_PROMPT="""You are an expert video understanding model with access to a video zoom tool that allows you to request high-frame-rate clips for temporal inspection. Your task is to correct a flawed analysis of a low-frame-rate video by using a `video_zoom` tool.
Your workflow is a multi-turn process:
**Turn 1: Reflection and Tool Call**
1.  **Analyze the Error**: You will be given a question, choices, and a previous, incorrect attempt. First, you must reflect on *why* the previous `video_zoom` tool call was flawed. Was the time segment wrong? Was the frames-per-second (fps) too low? Was the focus of the analysis misaligned with the question?
2.  **Formulate a Correction**: Based on your analysis, decide on a new, corrected `video_zoom` request. This request should target the precise moment of interest and use an appropriate fps to capture the fine-grained detail.
3.  **Output the Tool Call**: Generate your reflection and the new tool call in the specified format. **Your output for this turn MUST end immediately after the `</video_zoom>` tag.** Do not generate anything further. The system will then execute this call and provide you with the result.
**Constraint for the tool call**: The total number of frames requested must not exceed 16. That is: `(end_sec - start_sec) * fps <= 16`.
**Turn 2: Analysis and Final Answer**
1.  **Receive Tool Response**: The system will provide the high-frame-rate video clip from your corrected tool call.
2.  **Analyze the New Clip**: Carefully examine the new clip. Describe what you can now clearly see that resolves the question.
3.  **Provide the Final Answer**: Based on your new observation, state the correct answer from the choices, enclosed in `\boxed{}`.
---
**Output Format Structure:**
**FIRST TURN OUTPUT:**
<think>
The previous tool call was incorrect because [explain the flaw in the tool use, e.g., wrong segment, wrong fps, or misaligned focus].
Now I will zoom in to inspect the motion of '{target object/action}' between {start_sec}s and {end_sec}s with higher temporal resolution.
</think><video_zoom> {"segment": [start_sec, end_sec], "fps": n} </video_zoom>
**[YOUR TURN 1 OUTPUT STOPS HERE]**
---
**SECOND TURN OUTPUT (after you receive the tool response):**
<think>
In the corrected high-frame-rate clip, [describe what is clearly observed now].
</think>
<answer>
\\boxed{correct answer}
</answer>
---
**Example to follow:**
**Question:** Which hand did the woman use to pick up the cup?
**Choices:** A: Left hand B: Right hand C: Both hands D: Neither
**Previous Trajectory (Wrong):**
Tool call: `<video_zoom> {"segment": [0.0, 2.0], "fps": 2}</video_zoom>`
Answer: A (Left hand)
**(Your First Turn Output Should Look Like This):**
<think>
The previous tool call was incorrect because it focused on the wrong time segment. The woman only reaches for the cup between 3.0s and 5.0s. Additionally, the low fps of 2 might not be sufficient to clearly distinguish the hand's motion.
Now I will zoom in to inspect the motion of 'the woman's hand reaching for the cup' between 3.0s and 5.0s with a higher temporal resolution.
</think><video_zoom> {"segment": [3.0, 10.0], "fps": 1} </video_zoom>
**(System provides tool response, then you start your Second Turn)**
**(Your Second Turn Output Should Look Like This):**
<think>
In the corrected high-frame-rate clip, the woman's right hand is clearly seen moving towards and gripping the cup handle between 4.1s and 4.8s, while her left hand remains on her lap. The motion is now unambiguous.
</think>
<answer>
B.
</answer>"""
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
    stop=stop_after_attempt(3),  # 最大重试次数
    wait=wait_exponential(multiplier=1, min=1, max=8),  # 指数退避 (1, 2, 4, 8, ..., 128)
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
async def multiturn_query_client_with_tool(client, model, messages,video_path , semaphore=None, video_kwargs=None,timeout=60,storage_system='local',max_pixels=100352,min_pixels=25088,tool_call_max_frames=16, max_generation_round=2):
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
    # sft_messages[1]['content'] = sft_messages[1]['content'].split('Now answer the following question:')[-1]
    image_paths = image_paths[:]
    return sft_messages, image_paths


from qwen_vl_utils import process_vision_info
def prepare_message_for_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
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

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}
import uuid
async def evaluate(video_path ,item, semaphore, client, model, rollout_num: int = 1, fps=0.2, max_pixels=100352, pass_video_as_frames: bool = True, frames_upbound=30, timeout=60, use_fewshots=True, max_generation_round=2, save_sft=True, tool_call_max_frames=16):
    async with semaphore:  # 自动获取和释放信号量
        reflection_rollout = []
        extra_info = item['extra_info']
        problem = item['problem']
        answer = str(item['solution'])
        detailed_rollout = item['detailed_rollout']
        for rollout in detailed_rollout:
            sft_messages = copy.deepcopy(rollout['sft_messages'])
            if rollout['accuracy'] > 0 or len(sft_messages['messages']) <= 3:
                continue
            if pass_video_as_frames:
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{REFLECTION_PROMPT}",}
                        ]
                    },
                ]
                messages.append({   # gpt message
                    "role": "user",
                    "content": []
                    })
                # messages.extend(copy.deepcopy(rollout['sft_messages']['messages'][1:-1]))
                prompt_images = sft_messages['images']
                img_idx = 0
                try:
                    for message in sft_messages['messages'][1:-1]:
                        messages[-1]['content'].append({
                                "type": "text",
                                "text": f"{message['role']}: \n"
                        })
                        if message['role'] == 'user':
                            
                            text_splits = message['content'].split('<image>')
                            message['content'] = []
                            for i, text_split in enumerate(text_splits):
                                messages[-1]['content'].append({"type": "text", "text": text_split})
                                message['content'].append({"type": "text", "text": text_split})
                                if i < len(text_splits) - 1:
                                    # print(img_idx)
                                    img_data = read_img_as_base64(prompt_images[img_idx])
                                    messages[-1]['content'].append({"type": "image_url", "image_url": {"url": img_data}})
                                    message['content'].append({"type": "image_url", "image_url": {"url": img_data}})
                                    img_idx += 1
                        elif isinstance(message['content'], str):
                            messages[-1]['content'].append({"type": "text", "text": message['content']})
                            message['content'] = [{"type": "text", "text": message['content']}]
                    assert img_idx == len(prompt_images), f"img_idx={img_idx}, len(prompt_images)={len(prompt_images)}"
                    # assert img_idx <= 120, f"img_idx={img_idx} should be less than 120"
                    assert img_idx > 0, f"img_idx={img_idx} should be more than 0"
                except Exception as e:
                    logger.error(f"message error: {e}, img_idx={img_idx}, len(prompt_images)={len(prompt_images)}")
                    continue
            else:
                raise NotImplementedError
            # import pdb; pdb.set_trace()
            assert rollout_num == 1, "reflection only support rollout_num=1"
            for i in range(rollout_num):
                try:
                    pred, new_messages, round_n = await multiturn_query_client_with_tool(client, model, messages,video_path,timeout=timeout, storage_system='local', max_pixels=max_pixels, max_generation_round=max_generation_round, tool_call_max_frames=tool_call_max_frames)
                    reflection_message = sft_messages['messages'][:-1] + new_messages[2:] # 拼接原本错误轨迹和 gpt4o 纠正轨迹形成 reflection sft 轨迹
                    # import pdb; pdb.set_trace()
                    res = {}
                    if extra_info['answer_type'] == 'multiple_choice':
                        res = judge_multi_choice(problem, pred, answer)
                    elif extra_info['answer_type'] =='open_ended':
                        res = await judge_open_ended(problem, pred, answer)
                    else:
                        raise NotImplementedError
                    res['pred'] = pred
                    res['messages'] = extract_text_from_messages(new_messages) # gpt原始message
                    import time
                    if save_sft:
                        unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_path}_{problem}_{answer}_{rollout['rollout_index']}")
                        sft_res = extract_message_to_sft_format(reflection_message, f"/mnt/bn/tiktok-mm-5/aiic/users/dingyang/data/sft_images/{now}/{unique_id}", 
                        prompt_image_num=len(prompt_images), prompt_image_paths=prompt_images)
                        # import pdb; pdb.set_trace()

                        res['sft_messages'] = {
                            "messages": sft_res[0],
                            "images": sft_res[1]
                        }
                        logger.debug(f"{len(sft_res[1])}")
                    res['round'] = round_n
                    res['rollout_index'] = rollout['rollout_index']
                    reflection_rollout.append(res)
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    continue
        # pass_rate = sum([r['accuracy'] for r in detailed_res]) / (len(detailed_res) + 1e-5)
        # logger.debug(f'pass_rate: {pass_rate}')
        return {'reflection_rollout': reflection_rollout}


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
    # df = pd.read_json(args.json_path)
    # df.head()
    train_set = json.load(open(args.json_path))[:]
    semaphore = asyncio.Semaphore(args.max_concurrency)   # 限制并发数

    counter = defaultdict(int)
    round_counter = defaultdict(int)
    batch_size = args.batch_size if args.batch_size > 0 else len(train_set)
    num_batches = (len(train_set) + batch_size - 1) // batch_size
    base_name = os.path.basename(args.json_path).replace('.json', '')
    model_name = os.path.basename(args.model).replace('/', '_')
    save_dir = os.path.join(os.path.dirname(args.json_path), f"reflection_{model_name}_{args.name}")
    os.makedirs(save_dir, exist_ok=True)

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
                # counter[item['pass_rate']] += 1
                for rollout in item.get('reflection_rollout', []):
                    round_counter[rollout['round']] += 1
            continue

        tasks = []
        for item in batch_data:
            video_url = item['videos'][0]
            task = asyncio.create_task(evaluate(
                video_url, item, semaphore,
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
            batch_data[idx]['reflection_rollout'] = result['reflection_rollout']
            for item in result['reflection_rollout']:
                round_counter[item['round']] += 1
            all_results.append(batch_data[idx])

        # ✅ 保存一次中间结果
        if args.save_results and num_batches > 1:
            with open(batch_save_path, 'w') as f:
                json.dump(batch_data, f, indent=4)
            logger.info(f"Batch {batch_idx + 1} saved to {batch_save_path}")

    logger.info(len(all_results))
    # logger.info(f"pass rate: {counter}")
    logger.info(f"round_number: {round_counter}")

    if args.save_results:
        # ✅ 合并所有 batch 结果为最终输出
        final_save_path = os.path.join(
            save_dir,
            f'reflection_result_{model_name}.json'
        )
        with open(final_save_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Final merged results saved to {final_save_path}")

    return 

if __name__ == "__main__":
    asyncio.run(main())

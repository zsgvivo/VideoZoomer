from vllm import LLM, SamplingParams

# Specify the maximum number of frames per video to be 4. This can be changed.
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

# def main():
import sys
logger.add(sys.stdout, level="INFO")
# json_path = '/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/MathVision/oe_train_subproblem_gpt4o_verified.json'
# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Video Evaluation Script')

# 添加命令行参数
parser.add_argument('--json_path', type=str, default='/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/LLaVA-Video-178k/0_30_s_nextqa/0_30_s_nextqa_mc_qa_processed.json',
                    help='Path to the JSON file containing the dataset')
parser.add_argument('--fps', type=float, default=0.2,
                    help='Frames per second for video processing')
parser.add_argument('--max_pixels', type=int, default=100352,
                    help='Maximum number of pixels for resizing video frames')
parser.add_argument('--eval_rollout_num', type=int, default=1,
                    help='Number of evaluation rollouts per video')
parser.add_argument('--model', type=str, default='/mnt/bn/tiktok-mm-4/aiic/users/dingyang/models/Qwen2.5-VL-7B-Instruct',
                    help='model name to eval')
parser.add_argument('--batch_size', type=int, default=20,
                    help='Maximum number of concurrent tasks')

args = parser.parse_args()
llm = LLM(args.model, limit_mm_per_prompt={"image": 20}, tensor_parallel_size=4)



def _require_env(var_name: str) -> str:
    """Fetch an environment variable or raise if missing."""
    value = os.environ.get(var_name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return value


azure_openai_endpoint = _require_env("AZURE_OPENAI_VISION_ENDPOINT")
azure_openai_key = _require_env("AZURE_OPENAI_VISION_API_KEY")
azure_openai_version = os.environ.get("AZURE_OPENAI_VISION_API_VERSION", "2024-09-01-preview")
gptclient = openai.AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_version=azure_openai_version,
    api_key=azure_openai_key,
)
gptmodel = os.environ.get("AZURE_OPENAI_VISION_MODEL", "gpt-4o-2024-11-20")

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
    def extract_choice(answer):
        import re
        pattern = r"([A-E])\."
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
def judge_open_ended(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
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
    response = query_client_with_retry(gptclient, gptmodel, messages)
    logger.debug(f"gpt response: {response}")
    if len(response) == 0:
        score = 0
    else:
        score = 1 if '1' in response else 0
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
        raise NotImplementedError("Only local storage is supported for video loading.")
    vr = VideoReader(video_id_or_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]  # FPS Sampling
    frame_time = [i / avg_fps for i in frame_idx]

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

async def process_video_async(video_id_or_path: str, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512, storage_system: str = 'local', video_fps=0.2, frames_upbound=30):
    # 异步执行IO密集型操作
    def sync_video_processing():
        # 这部分是同步操作，但将在线程池中运行
        if storage_system != 'local':
            raise NotImplementedError("Only local storage is supported for video loading.")
        vr = VideoReader(video_id_or_path, ctx=cpu(0), num_threads=1)
        
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        avg_fps = round(vr.get_avg_fps() / video_fps)
        frame_idx = [i for i in range(0, total_frame_num, avg_fps)]  # FPS Sampling
        frame_time = [i / avg_fps for i in frame_idx]

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
    
    # 在事件循环的默认线程池中执行IO密集型操作
    return await asyncio.to_thread(sync_video_processing)

logger_std = logging.getLogger(__name__)
@retry(
    stop=stop_after_attempt(20),  # 最大重试次数
    wait=wait_exponential(multiplier=1, min=1, max=128),  # 指数退避 (1, 2, 4, 8, ..., 128)
    retry=(retry_if_exception_type(Exception)),  # 对所有异常重试
    before_sleep=before_sleep_log(logger_std, logging.WARNING),  # 重试前记录日志
    reraise=True  # 重试结束后重新抛出异常
)
async def query_async_client_with_retry(client, model, messages, semaphore=None):
    chat_response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=1.0,
                )
    response_str = chat_response.choices[0].message.content
    return response_str

from retrying import retry
def log_retry_attempt(exception):
    """在重试时记录日志"""
    logger.warning(f"Retrying due to exception: {str(exception)}")
    return True  # 继续重试
@retry(
    stop_max_attempt_number=20,  # 最大重试次数
    wait_exponential_multiplier=1000,  # 初始等待时间 (毫秒)
    wait_exponential_max=128000,  # 最大等待时间 (128秒)
    retry_on_exception=lambda exc: isinstance(exc, Exception),  # 对所有异常重试
    before_sleep=log_retry_attempt  # 重试前记录日志
)
def query_client_with_retry(client, model, messages, semaphore=None):
    chat_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
    )
    response_str = chat_response.choices[0].message.content
    return response_str
def build_message(video_path, problem, answer, extra_info, rollout_num: int = 1, fps=0.2, max_pixels=100352):
    detailed_res = []
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.\nOutput the thinking process within <think> </think> tags and final answer within <answer> </answer> tags. The final answer should contain \\boxed{{}}",
                }
            ]
        },
        {
        "role": "user",
        "content": [],
    }]
    video_frames = process_video(video_path, max_pixels=max_pixels, video_fps=fps)
    for idx, frame in enumerate(video_frames):
        messages[1]['content'].append({
            "type": "text",
            "text": f"<frame{idx}>"
        })
        messages[1]['content'].append({
            "type": "image_url",
            "image_url": {
                "url": read_img_as_base64(frame),
            }
        }) 
    messages[1]['content'].append({
                "type": "text",
                "text": problem, 
            })
    return messages
        

sampling_params = SamplingParams(temperature=1.0, max_tokens=512)
df = pd.read_json(args.json_path)
df.head()
train_set = df.to_dict(orient='records')


counter = defaultdict(int)
batch_size = args.batch_size
from concurrent.futures import ThreadPoolExecutor
def process_item(item):
    return build_message(
        item['videos'][0], 
        item['problem'], 
        str(item['solution']), 
        item['extra_info'], 
        args.eval_rollout_num, 
        args.fps, 
        args.max_pixels
    )
# for problem_id, item in enumerate(train_set[:]):
for i in range(0, len(train_set), batch_size):
    batch = train_set[i:i+batch_size]
    messages = []
    # for problem_id, item in enumerate(batch):
    with ThreadPoolExecutor() as executor:
        messages = list(executor.map(process_item, [item for _, item in enumerate(batch)]))
    outputs = llm.chat(messages,
                sampling_params=sampling_params,
                use_tqdm=True)
    for idx, output in enumerate(outputs):
        item = batch[idx]
        train_set[i+idx]['detailed_rollout'] = {output.outputs[0].text}
        if train_set[i+idx]['extra_info']['answer_type'] == 'multiple_choice':
            train_set[i+idx]['pass_rate'] = judge_multi_choice(item['problem'], output.outputs[0].text, item['solution'])['accuracy']
        elif train_set[i+idx]['extra_info']['answer_type'] =='open_ended':
            train_set[i+idx]['pass_rate'] = judge_open_ended(item['problem'], output.outputs[0].text, item['solution'])['accuracy']
        counter[train_set[i+idx]['pass_rate']] += 1
# logger.info(len(results))
logger.info(counter)

# Define the new save directory
save_dir = os.path.join(os.path.dirname(args.json_path), 'eval')

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Construct the new save path with eval_rollout_num
base_name = os.path.basename(args.json_path).replace('.json', '')
model_name = os.path.basename(args.model).replace('/', '_')
save_path = os.path.join(save_dir, f'{base_name}_model{model_name}_eval_rollout{args.eval_rollout_num}_fps{args.fps}_max_pixels{args.max_pixels}.json')
with open(save_path, 'w') as f:
    json.dump(train_set, f, indent=4)
logger.info(f'results saved to {save_path}')
# save pass rate
save_dir2 = os.path.join(save_dir, f'{base_name}_model{model_name}_eval_rollout{args.eval_rollout_num}_fps{args.fps}_max_pixels{args.max_pixels}_passrate.json')
with open(save_dir2, 'w') as f:
    json.dump(counter, f, indent=4)
logger.info(f'pass rate saved to {save_dir2}')


# if __name__ == "__main__":
    # asyncio.run(main())
    # main()

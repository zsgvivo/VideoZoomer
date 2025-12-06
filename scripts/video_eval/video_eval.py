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

def read_img_as_base64(img):
    if isinstance(img, str):
        pil_img = Image.open(img)
    else:
        pil_img = img
    buffered = BytesIO()
    format = "PNG" if pil_img.mode in ("RGBA", "LA") else "JPEG"
    pil_img.save(buffered, format=format)
    return f"data:image/{format.lower()};base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

# def math_acc_reward(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
#     accuracy_reward = 0.0
#     # math_verify
#     try:
#         parsed_ground_truth = parse(ground_truth)
#         parsed_predict_str = parse(predict_str, extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()])
#         if float(verify(parsed_ground_truth, parsed_predict_str)) > 0:
#             accuracy_reward = 1.0
#     except Exception as e:
#         logger.info(f"math_verify failed: {e}. Parsed Model Prediction: {parsed_predict_str}, Parsed Ground Truth: {parsed_ground_truth}")
#         pass  # Continue to next verification method if this fails
#     # mathruler
#     if accuracy_reward == 0.0:
#         try:
#             extracted_answer = extract_boxed_content(predict_str)
#             if grade_answer(extracted_answer, ground_truth):
#                 accuracy_reward = 1.0
#         except Exception as e:
#             logger.info(f"mathruler failed: {e}. Extracted Answer: {extracted_answer}, Ground Truth: {ground_truth}")
#             pass
#     accuracy_reward_dict = {
#         "accuracy": accuracy_reward,
#     }
#     return accuracy_reward_dict

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
import os
from decord import VideoReader, cpu
import io

def process_video(video_id_or_path: str, max_pixels: int = 2048 * 2048, min_pixels: int = 25088, storage_system: str = 'local', video_fps=0.2, frames_upbound=30):
    from io import BytesIO
    import cv2
    from PIL import Image
    import numpy as np
    
    # Currently, we do not support load precomputed features
    if storage_system != 'local':
        raise NotImplementedError("Only local storage is supported for video loading.")
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
                raise NotImplementedError("Only local storage is supported for video loading.")
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
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return []
    
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

async def evaluate(video_path, problem, answer, extra_info, semaphore, client, model, rollout_num: int = 1, fps=0.2, max_pixels=100352, pass_video_as_frames: bool = True, frames_upbound=30, timeout=60):
    async with semaphore:  # 自动获取和释放信号量
        detailed_res = []
        if pass_video_as_frames:
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
            if fps > 0:
                video_frames = await process_video_async(video_path, max_pixels=max_pixels, video_fps=fps, frames_upbound=frames_upbound)
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
            video_kwargs = None
        else:
            raise NotImplementedError
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
                "content": [
                    {
                        "type": "video", 
                        "video" :"file://" + video_path,
                        "max_pixels": max_pixels,
                        "fps": fps
                    },
                    {"type": "text", "text": problem}
                ],
            }]
            messages, video_kwargs = prepare_message_for_vllm(messages)
        preds = await query_client_with_retry(client, model, messages, video_kwargs, rollout_num=rollout_num,timeout=timeout)
        for i, pred in enumerate(preds):
            res = {}
            if extra_info['answer_type'] == 'multiple_choice':
                res = judge_multi_choice(problem, pred, answer)
            elif extra_info['answer_type'] =='open_ended':
                res = await judge_open_ended(problem, pred, answer)
            else:
                raise NotImplementedError
            res['pred'] = pred
            res['rollout_index'] = i
            detailed_res.append(res)
        pass_rate = sum([r['accuracy'] for r in detailed_res]) / len(detailed_res)
        logger.debug(f'pass_rate: {pass_rate}')
        return {'pass_rate': pass_rate, 'detailed_rollout': detailed_res}


async def main():
    parser = argparse.ArgumentParser(description='Video Evaluation Script')

    parser.add_argument('--json_path', type=str,
                        default='/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/LLaVA-Video-178K/0_30_s_nextqa/0_30_s_nextqa_mc_qa_processed.json')
    parser.add_argument('--fps', type=float, default=0.2)
    parser.add_argument('--max_pixels', type=int, default=100352)
    parser.add_argument('--eval_rollout_num', type=int, default=1)
    parser.add_argument('--model', type=str,
                        default='/mnt/bn/tiktok-mm-4/aiic/users/dingyang/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--openai_api_base', type=str, default='http://localhost:8000/v1')
    parser.add_argument('--max_concurrency', type=int, default=20)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--pass_video_as_frames', action='store_true')
    parser.add_argument('--frames_upbound', type=int, default=120)
    parser.add_argument('--request_timeout', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=-1,
                        help='batch size for data processing; -1 means one batch')
    args = parser.parse_args()

    # ---------------- 客户端初始化 ----------------
    if args.model == gptmodel:
        client = gptclient
        logger.info("using gpt model")
    elif args.model == geminimodel:
        if geminiclient is None:
            raise EnvironmentError("Gemini client is not configured. Set AZURE_GEMINI_* environment variables.")
        client = geminiclient
        logger.info("using gemini model")
    else:
        client = openai.AsyncOpenAI(api_key="EMPTY", base_url=args.openai_api_base)
    # ---------------------------------------------

    # 读数据
    df = pd.read_json(args.json_path)
    train_set = df.to_dict(orient='records')

    # 保存路径
    base_name = os.path.basename(args.json_path).replace('.json', '')
    model_name = os.path.basename(args.model).replace('/', '_')
    save_dir = os.path.join(os.path.dirname(args.json_path), 'eval',
                            f"{base_name}_model{model_name}_eval_rollout{args.eval_rollout_num}"
                            f"_fps{args.fps}_max_pixels{args.max_pixels}")
    os.makedirs(save_dir, exist_ok=True)

    # batch 相关
    batch_size = args.batch_size if args.batch_size > 0 else len(train_set)
    num_batches = (len(train_set) + batch_size - 1) // batch_size

    semaphore = asyncio.Semaphore(args.max_concurrency)

    all_results = []
    counter = defaultdict(int)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(train_set))
        batch_data = train_set[start:end]

        batch_save_path = os.path.join(save_dir, f'batch_{batch_idx + 1}_of_{num_batches}.json')
        if os.path.exists(batch_save_path):
            logger.info(f"Batch {batch_idx + 1} already processed. Skipping.")
            # 把旧结果读回来，用于最后 merge
            with open(batch_save_path, 'r') as f:
                batch_results = json.load(f)
            all_results.extend(batch_results)
            for item in batch_results:
                counter[item['pass_rate']] += 1
            continue

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({start}-{end})")
        tasks = [
            asyncio.create_task(evaluate(
                item['videos'][0],
                item['problem'],
                str(item['solution']),
                item['extra_info'],
                semaphore,
                client,
                args.model,
                args.eval_rollout_num,
                args.fps,
                args.max_pixels,
                pass_video_as_frames=args.pass_video_as_frames,
                frames_upbound=args.frames_upbound,
                timeout=args.request_timeout
            ))
            for item in batch_data
        ]

        results = await tqdm_asyncio.gather(*tasks)

        # 写回 batch_data
        for idx, res in enumerate(results):
            batch_data[idx]['pass_rate'] = res['pass_rate']
            batch_data[idx]['detailed_rollout'] = res['detailed_rollout']
            counter[res['pass_rate']] += 1

        all_results.extend(batch_data)

        # 保存中间 batch
        if args.save_results and num_batches > 1:
            with open(batch_save_path, 'w') as f:
                json.dump(batch_data, f, indent=4)
            logger.info(f"Batch {batch_idx + 1} saved to {batch_save_path}")

    logger.info(f"Total results: {len(all_results)}")
    logger.info(f"pass rate: {dict(counter)}")

    if args.save_results:
        # 合并结果
        final_path = os.path.join(save_dir, 'merged_result.json')
        with open(final_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Final merged results saved to {final_path}")

        # 保存统计
        stat_path = os.path.join(save_dir, 'passrate.json')
        with open(stat_path, 'w') as f:
            json.dump(counter, f, indent=4)
        logger.info(f"pass rate saved to {stat_path}")

if __name__ == "__main__":
    asyncio.run(main())

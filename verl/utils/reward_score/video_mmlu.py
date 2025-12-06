import re
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
import time
import openai
import random
import requests
from PIL import Image
import io
import base64
from PIL import Image
from .toolcall_format import format_reward
from .tool_call_extra_reward import tool_call_extra_score

# use prompt from https://github.com/Espere-1119-Song/Video-MMLU/blob/main/post_eval/eval_reason_qa.py
SYSTEM_PROMPT = """
You are an intelligent chatbot designed for evaluating the correctness of generative outputs for reasoning-based question-answer pairs. 
Your task is to compare the predicted answer with the correct answer based on the following rules:
------
## INSTRUCTIONS:
1. **Evaluate Reasoning Tasks Strictly:**
   - The predicted answer must capture all critical concepts and details mentioned in the correct answer. 
   - If the correct answer mentions specific concepts or examples (e.g., 'odd numbers accumulate to form perfect squares'), the predicted answer must include these concepts or examples. 
   - Even if the phrasing differs, the key meaning and concepts must be preserved. However, omitting or altering key concepts or examples is **not acceptable**.
   - **Example 1:** If the correct answer is 'The construction method shows how odd numbers accumulate to form perfect squares,' the redicted answer must include 'odd numbers' and 'perfect squares.'
   - **Example 2:** If the correct answer is 'To eliminate HBr and form an alkene,' the predicted answer must address the elimination of HBr as well.
   - Minor differences in phrasing are acceptable as long as the key information is retained.
   - **Critical Detail:** If any essential element (e.g., key terms, concepts, or examples) is missing from the predicted answer, the answer is considered incorrect.
   - Do **not** introduce new, unrelated information in the predicted answer.
"""

QUERY_PROMPT = """
Please evaluate the following reasoning-based question-answer pair:\n\n
Question: {question}\n
Correct Answer: {ground_truth}\n
Predicted Answer: {prediction}\n\n
Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. 
Ensure that the predicted answer captures all critical concepts and details from the correct answer, without omitting key elements. 
Minor rewording is acceptable, but the meaning and essential details must remain the same. 
If the predicted answer misses any critical concept or introduces unrelated information, it should be judged as incorrect. 
Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING.
DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. 
For example, your response should look like this: {{"pred": "no", "score": 3}}
"""



def get_image_data_url(image_input):
    if isinstance(image_input, str) and image_input.startswith("data:"):
        return image_input

    if isinstance(image_input, str):
        image_input = Image.open(image_input)

    if not isinstance(image_input, Image.Image):
        raise ValueError("Unsupported image input type")

    if image_input.mode != "RGB":
        image_input = image_input.convert("RGB")

    buffer = io.BytesIO()
    image_input.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    base64_data = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"

class GPT4VisionClient:
    """Client for interacting with GPT-4 Vision API."""

    def __init__(self, endpoint=None, api_key=None, model=None):
        base_url = endpoint or os.environ.get("VIDEO_MMLU_API_BASE", "http://localhost:8000/v1")
        api_key = api_key or os.environ.get("VIDEO_MMLU_API_KEY", "sk-no-key-required")
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model or os.environ.get("VIDEO_MMLU_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    def query(
        self, images, prompt: str, system_prompt: str = None, max_retries=3, initial_delay=3
    ) -> str:
        """Query GPT-4 Vision with an image and prompt"""
        # if images is None:
        #     return None
        delay = initial_delay
        data_url_list = []
        for image in images:
            data_url_list.append(
                get_image_data_url(image)
            )  # Assuming this function exists

        if system_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                },
            ]
        else:
            messages = []
        messages.append(
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": prompt},
                    # {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        )

        for data_url in data_url_list:
            messages[-1]["content"].append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )

        messages[-1]["content"].append({"type": "text", "text": prompt})

        # import pdb; pdb.set_trace()

        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=min(0.2*attempt, 1.0),
                    max_tokens=1024,
                    timeout=5,
                )

                return response.choices[0].message.content

            except Exception as e:
                print("="*100)
                print(str(e))
                print("messages: ", messages)
                print("="*100)
                delay *= 2
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            attempt += 1
        print(f"Warning: Failed after {max_retries} attempts")
        return ""

client = GPT4VisionClient()

# def format_reward(predict_str: str, extra_info: dict = None) -> float:
#     pattern = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL | re.MULTILINE)
#     match_result = re.fullmatch(pattern, predict_str)
#     return 1.0 if match_result else 0.0

import ast
def inner_acc_reward(prompt:str, predict_str: str, original_answer: str, use_gpt=False, gpt_extract_answer=False):

    original_predict_str = predict_str
    
    if gpt_extract_answer:
        original_predict_str = original_predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    question = prompt

    # print("="*100)
    # print("use_gpt: {}, gpt_extract_answer: {}".format(use_gpt, gpt_extract_answer))
    # print("question: {}".format(question))
    # print("original_answer: {}".format(original_answer))
    # print("prediction: {}".format(original_predict_str))
    # print("="*100)

    # import pdb; pdb.set_trace()

    prompt = QUERY_PROMPT.format(question=question, ground_truth=original_answer, prediction=original_predict_str)
    response = client.query(images=[], prompt=prompt, system_prompt=SYSTEM_PROMPT)
    # if len(response) == 0:
    #     reward = {"is_filter": True, "info": "error with gpt4o"}
    # else:
    #     reward = 1.0 if '1' in response else 0.0
    try:
        judgement_string = response
        judgement_dict = ast.literal_eval(judgement_string)
    except Exception as e:
        print(f"Reward Error: {e}")
        judgement_dict = {'pred': 'no', 'score': 0}

    reward = judgement_dict['score']
    return reward


def acc_reward(prompt: str, predict_str: str, solution: str, extra_info: dict = None) -> float:
    gpt_extract_answer = extra_info.get("gpt_extract_answer", False)
    reward = inner_acc_reward(prompt, predict_str, solution, use_gpt=True, gpt_extract_answer=gpt_extract_answer)
    return reward

def compute_score(prompt: str, predict_str: list, ground_truth: list, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    tool_call_penalty = extra_info['tool_call_penalty']
    forced_tool_call = extra_info['forced_tool_call']
    filter_unfinished_traj = extra_info['filter_unfinished_traj']
    if filter_unfinished_traj:
        if predict_str[-1].count('<video_zoom>') > 0 and predict_str[-1].count('<answer>') == 0:
            return {"is_filter": True, "info": "unfinished trajactory"}
    format, tool_call_count, tool_call_frame_num = format_reward(predict_str, extra_info)
    if forced_tool_call and tool_call_count == 0:
        return 0, 0, 0
    acc = acc_reward(prompt, ' '.join(predict_str), ground_truth, extra_info)
    
    if isinstance(acc, dict):
        return acc
    # print(f"acc_reward_weight: {acc_reward_weight}, format_reward_weight: {format_reward_weight}")
    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format
    score = acc_score + format_score
    # if penalize_all_tool_call:
    #     if tool_call_count > 0:
    #         score -= tool_call_penalty
    # else:
    #     if acc == 1 and tool_call_count > 0:
    #         score -= tool_call_penalty
    extra_score = tool_call_extra_score(tool_call_count, tool_call_frame_num, acc, extra_info)
    score += extra_score

    # print(f"score: {score}, acc_score: {acc_score}, format_score: {format_score}")

    # print("="*100 + "\n" + "prompt: " + prompt + "\n" + "predict_str: " + predict_str + "\n" + "ground_truth: " + ground_truth + "\n" + "score: " + str(score) + "\n" + "acc_score: " + str(acc_score) + "\n" + "format_score: " + str(format_score) + "\n" + "="*100)

    return score, acc_score, format_score


if __name__ == '__main__':
    question = "Elena Ferrante" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = ["""<think>\nTo answer the question, I will locate Kevin Watson\'s transfer in the table and identify the fee information associated with it.\n</think>\n<video_zoom></video_zoom>""", """<think>\nI think.\n</think>\n<answer>\n0\n</answer>"""]
    ground_truth = "0" #"Martha White" #"china" #"$ 2 $" #"A" #"1:3" #"0.5 cm" #"0.5"
    extra_info = {
        "acc_reward_weight": 0.95,
        "format_reward_weight": 0.05,
        "tool_call_penalty": -0.5,
        "forced_tool_call": False,
    }
    s1 = compute_score(question, predict_str, ground_truth, extra_info)
    print(s1)

    s2 = format_reward(predict_str, extra_info)
    print(s2)

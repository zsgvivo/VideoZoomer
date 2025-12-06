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

SYSTEM_PROMPT = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\nYour task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n- Focus on the meaningful match between the predicted answer and the correct answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the correctness of the prediction compared to the answer."

QUERY_PROMPT = """I will give you a question related to an image and the following text as inputs:\n\n1. **Question Related to the Image**: {question}\n2. **Ground Truth Answer**: {ground_truth}\n3. **Model Predicted Answer**: {prediction}\n\nYour task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?\n- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:\n(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.\n(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct.\n**Output Format**:\nYour response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model's prediction strictly aligns with the ground truth, while 0 means it does not.\nThe format should be \"Score: 0 or 1\""""

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

    def __init__(self, endpoint=None, api_key=None, api_version=None, model=None):
        endpoint = endpoint or os.environ.get("AZURE_OPENAI_VISION_ENDPOINT")
        api_key = api_key or os.environ.get("AZURE_OPENAI_VISION_API_KEY")
        api_version = api_version or os.environ.get("AZURE_OPENAI_VISION_API_VERSION", "2024-09-01-preview")
        if not endpoint or not api_key:
            raise EnvironmentError(
                "Missing Azure OpenAI Vision credentials. "
                "Set AZURE_OPENAI_VISION_ENDPOINT and AZURE_OPENAI_VISION_API_KEY."
            )
        self.client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        )
        self.model = model or os.environ.get("AZURE_OPENAI_VISION_MODEL", "gpt-4o-2024-11-20")

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
                        {"type": "text", "text": system_prompt},
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
    if 'score:' not in response.lower():
        # raise ValueError(f"No 'score:' in response: {response}")
        reward = {"is_filter": True, "info": "error with gpt4o"}
        print("Extraction error, response: ", response)
    extract_score = response.lower().split("score:")[-1].strip().split("\n")[0].strip().split(" ")[0]
    if "1" not in extract_score and '0' not in extract_score:
        reward = {"is_filter": True, "info": "error with gpt4o"}
        print("Extraction error, response: ", response)
    else:
        reward = 1.0 if '1' in extract_score else 0.0

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

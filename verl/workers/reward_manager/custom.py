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


import re
import torch
import numpy as np
from verl import DataProto
from transformers import PreTrainedTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
from verl.trainer.constants import OPEN_ENDED_DATA_SOURCES

### Easy-R1 Math Reward ###
def math_format_reward(predict_str: str) -> dict:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    format_reward = 1.0 if format_match else 0.0
    format_reward_dict = {
        "format": format_reward,
    }
    return format_reward_dict


def math_acc_reward(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
    accuracy_reward = 0.0
    # math_verify
    try:
        parsed_ground_truth = parse(ground_truth)
        parsed_predict_str = parse(predict_str, extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()])
        if float(verify(parsed_ground_truth, parsed_predict_str)) > 0:
            accuracy_reward = 1.0
    except Exception as e:
        print(f"math_verify failed: {e}. Parsed Model Prediction: {parsed_predict_str}, Parsed Ground Truth: {parsed_ground_truth}")
        pass  # Continue to next verification method if this fails
    # mathruler
    if accuracy_reward == 0.0:
        try:
            extracted_answer = extract_boxed_content(predict_str)
            if grade_answer(extracted_answer, ground_truth):
                accuracy_reward = 1.0
        except Exception as e:
            print(f"mathruler failed: {e}. Extracted Answer: {extracted_answer}, Ground Truth: {ground_truth}")
            pass
    accuracy_reward_dict = {
        "accuracy": accuracy_reward,
    }
    return accuracy_reward_dict


def math_compute_score(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
    score_dict = {}
    score_dict.update(math_acc_reward(prompt_str, predict_str, ground_truth))
    score_dict.update(r1v_format_reward(predict_str))
    return score_dict

### R1-V Reward ###
def r1v_format_reward(predict_str: str) -> dict:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    format_reward = 1.0 if format_match else 0.0
    format_reward_dict = {
        "format": format_reward,
    }
    return format_reward_dict

def r1v_accuracy_reward(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
    accuracy_reward = 0.0
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if grade_answer(given_answer, ground_truth):
            accuracy_reward = 1.0
    except Exception:
        pass
    accuracy_reward_dict = {
        "accuracy": accuracy_reward,
    }
    return accuracy_reward_dict

def r1v_compute_score(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
    score_dict = {}
    score_dict.update(r1v_accuracy_reward(prompt_str, predict_str, ground_truth))
    score_dict.update(r1v_format_reward(predict_str))
    return score_dict

### MMRL Reward ###
def mmrl_accuracy_reward(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
    from verl.utils.reward_score.gpt import match_by_gpt4o
    accuracy_reward = 0.0
    try:
        question_match = re.search(r'user\n(.*?)\nassistant', prompt_str, re.DOTALL)
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if question_match and answer_match:
            question_str = question_match.group(1).strip()
            answer_str = answer_match.group(1).strip()
            accuracy_reward = match_by_gpt4o(question_str, answer_str, ground_truth)
            if accuracy_reward == None:
                accuracy_reward = 0.0
        else:
            accuracy_reward = 0.0
    except Exception:
        print("GPT4o Evaluation Failed, set accuracy reward to 0.")
        pass
    accuracy_reward_dict = {
        "accuracy": accuracy_reward,
    }
    return accuracy_reward_dict

def mmrl_compute_score(prompt_str: str, predict_str: str, ground_truth: str) -> dict:
    score_dict = {}
    score_dict.update(mmrl_accuracy_reward(prompt_str, predict_str, ground_truth))
    score_dict.update(r1v_format_reward(predict_str))
    return score_dict
### mv_math reward
def mv_math_accuracy_reward(data_source: str, prompt_str: str, predict_str: str, ground_truth: str, answer_type: str) -> dict:
    from verl.utils.reward_score.gpt import match_by_gpt4o
    accuracy_reward = 0.0
    try:
        question_match = re.search(r'user\n(.*?)\nassistant', prompt_str, re.DOTALL)
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if question_match and answer_match:
            question_str = question_match.group(1).strip()
            answer_str = answer_match.group(1).strip()
            if data_source == 'mvmath':
                prompt_type = data_source + '_' + answer_type
            else:
                prompt_type = 'default'
            accuracy_reward = match_by_gpt4o(question_str, answer_str, ground_truth, prompt_type)
            if accuracy_reward == None:
                accuracy_reward = 0.0
        else:
            print(f"Failed to extract answer")
            accuracy_reward = 0.0
    except Exception as e:
        print (f"GPT4o Evaluation Failed, set accuracy reward to 0. Error: {e}")
        
        pass
    accuracy_reward_dict = {
        "accuracy": accuracy_reward,
    }
    return accuracy_reward_dict

def mv_math_compute_score(data_source: str, prompt_str: str, predict_str: str, ground_truth: str, answer_type: str) -> dict:
    score_dict = {}
    score_dict.update(mv_math_accuracy_reward(data_source, prompt_str, predict_str, ground_truth, answer_type))
    score_dict.update(r1v_format_reward(predict_str))
    return score_dict

def overlength_reward(predict_str: str, max_resp_len=16384, overlong_buffer_cfg: dict = {"len": 4096, "penalty_factor": 1.0}) -> dict:
    """
    Calculate a reward penalty for responses that exceed the expected length.
    
    Args:
        predict_str: The prediction string to evaluate
        max_resp_len: Maximum allowed response length
        overlong_buffer_cfg: Configuration for overlength penalties
    
    Returns:
        Dictionary containing the overlength reward value
    """
    overlength_reward_value = 0.0
    
    response_length = len(predict_str.split())
    
    overlong_buffer_len = overlong_buffer_cfg.get("len", 4096)
    expected_len = max_resp_len - overlong_buffer_len
    exceed_len = response_length - expected_len
    
    if exceed_len > 0:
        overlong_penalty_factor = overlong_buffer_cfg.get("penalty_factor", 1.0)
        overlength_reward_value = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    
    return {
        "overlength": overlength_reward_value
    }

def _auto_compute_score(data_source: str, prompt_str: str, predict_str: str, ground_truth: str) -> float:
    # Training: Rule-based
    if data_source in ['text_only', 'MMPR', 'K12',]:
        score_dict = math_compute_score(prompt_str, predict_str, ground_truth)
    # Training: Open-ended QA
    # elif data_source in OPEN_ENDED_DATA_SOURCES:
    #     score_dict = mmrl_compute_score(prompt_str, predict_str, ground_truth)
    # Validation
    elif data_source in ['mathvista_testmini', 'mathvision_testmini',]:
        score_dict = math_acc_reward(prompt_str, predict_str, ground_truth)
    else:
        raise NotImplementedError()
    return score_dict

class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str, mode: str = "train", num_threads: int = 64, acc_reward_weight = 0.9, format_reward_weight = 0.1,  overlength_reward_weight: float = 0):

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score_type = compute_score
        self.mode = mode
        self.num_threads = num_threads

        self.reward_weight_dict = {
            "train": {
                "accuracy": acc_reward_weight / (format_reward_weight + acc_reward_weight + overlength_reward_weight),
                "format": format_reward_weight / (format_reward_weight + acc_reward_weight + overlength_reward_weight),
                "overlength": overlength_reward_weight / (format_reward_weight + acc_reward_weight + overlength_reward_weight),
            },
            "val": {
                "accuracy": 1.0,
                "format": 0.0,
                "overlength": 0.0,
            },
        }
        print(f"reward_weight_dict: {self.reward_weight_dict}")

        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == "mmrl":
            from concurrent.futures import ThreadPoolExecutor
            self.compute_score = mmrl_compute_score
            self.executor = ThreadPoolExecutor(num_threads)
        elif compute_score == "mmrl_acc":
            from concurrent.futures import ThreadPoolExecutor
            self.compute_score = mmrl_accuracy_reward
            self.executor = ThreadPoolExecutor(num_threads)
        elif compute_score == "mm_eureka":
            self.compute_score = math_compute_score
        elif compute_score == "mv_math":
            from concurrent.futures import ThreadPoolExecutor
            self.compute_score = mv_math_compute_score
            self.executor = ThreadPoolExecutor(num_threads)
        elif compute_score == "auto":
            from concurrent.futures import ThreadPoolExecutor
            self.compute_score = _auto_compute_score
            self.executor = ThreadPoolExecutor(num_threads)
        else:
            raise NotImplementedError()

    def __call__(self, data: DataProto) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        acc_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        format_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        prompt_str_list = []
        response_str_list = []
        ground_truth_list = []
        valid_response_length_list = []
        score_dict_list = []
        already_print = 0

        if self.compute_score_type in ["mmrl", "mmrl_acc", "auto", "mv_math"]:
            futures = []    # gpt4o evaluation

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["ground_truth"]

            prompt_str_list.append(prompt_str)
            response_str_list.append(response_str)
            ground_truth_list.append(ground_truth)
            valid_response_length_list.append(valid_response_length)

            if self.compute_score_type in ["math", "r1v", "mm_eureka"]:
                score_dict = self.compute_score(prompt_str, response_str, ground_truth)
                score_dict_list.append(score_dict)
                score = self.compute_score_from_dict(score_dict)
                reward_tensor[i, valid_response_length - 1] = score
                acc_reward_tensor[i, valid_response_length - 1] = score_dict["accuracy"]
                format_reward_tensor[i, valid_response_length - 1] = score_dict["format"]
                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    print("[score]", score)

            elif self.compute_score_type in ["mmrl", "mmrl_acc", "auto",]:
                if self.compute_score_type == "auto":
                    data_source = data_item.non_tensor_batch['data_source']
                    futures.append(self.executor.submit(self.compute_score, data_source, prompt_str, response_str, ground_truth))
                else:
                    futures.append(self.executor.submit(self.compute_score, prompt_str, response_str, ground_truth))
            elif self.compute_score_type == "mv_math":
                data_source = data_item.non_tensor_batch['data_source']
                futures.append(self.executor.submit(self.compute_score, data_source, prompt_str, response_str, ground_truth, data_item.non_tensor_batch['extra_info']['answer_type']))
        if self.compute_score_type in ["mmrl", "mmrl_acc", "auto", "mv_math"]:
            score_dict_list = [f.result() for f in futures]
            scores = torch.tensor([self.compute_score_from_dict(score_dict) for score_dict in score_dict_list])
            acc_scores = torch.tensor([score_dict["accuracy"] for score_dict in score_dict_list])
            format_scores = torch.tensor([score_dict["format"] for score_dict in score_dict_list])
            valid_response_lengths = torch.stack(valid_response_length_list) - 1    # -1 to act as idx
            reward_tensor.scatter_(1, valid_response_lengths.unsqueeze(1), scores.unsqueeze(1))
            acc_reward_tensor.scatter_(1, valid_response_lengths.unsqueeze(1), acc_scores.unsqueeze(1))
            format_reward_tensor.scatter_(1, valid_response_lengths.unsqueeze(1), format_scores.unsqueeze(1))
            while already_print < self.num_examine:
                print("[prompt]", prompt_str_list[already_print])
                print("[response]", response_str_list[already_print])
                print("[ground_truth]", ground_truth_list[already_print])
                print("[score]", scores[already_print].item())
                already_print += 1

        # metrics = self.update_metrics(metrics, score_dict_list, response_str_list)

        return reward_tensor, acc_reward_tensor, format_reward_tensor

    def compute_score_from_dict(self, score_dict: dict) -> float:
        score = 0.0
        reward_weight_dict = self.reward_weight_dict[self.mode]
        for reward_type in score_dict.keys():
            score += reward_weight_dict[reward_type] * score_dict[reward_type]
        return score

    def count_keywords_in_list(self, string_list):
        keyword_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.keyword_set)) + r')\b', re.IGNORECASE)
        total_count = sum(len(keyword_pattern.findall(string)) for string in string_list)
        return total_count

    def update_metrics(self, metrics: dict, score_dict_list: dict, response_str_list: list) -> dict:
        for reward_type in score_dict_list[0].keys():
            reward_list = [score_dict[reward_type] for score_dict in score_dict_list]
            metrics[f"{self.mode}/{reward_type}_mean"] = np.mean(reward_list)
        metrics[f"{self.mode}/self_reflective_indicator_count"] = self.count_keywords_in_list(response_str_list)
        return metrics
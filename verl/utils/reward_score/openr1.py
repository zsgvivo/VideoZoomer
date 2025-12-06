import re
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
from .toolcall_format import format_reward
from .tool_call_extra_reward import tool_call_extra_score

# https://github.com/Deep-Agent/R1-V/blob/main/src/r1-v/src/open_r1/grpo.py
# def format_reward(predict_str: str) -> float:
#     pattern = re.compile(r"\s*<think>.*?</think>.*<answer>.*?</answer>\s*", re.DOTALL | re.MULTILINE)
#     match_result = re.fullmatch(pattern, predict_str)
#     return 1.0 if match_result else 0.0

def math_acc_reward(predict_str: str, sol: str) -> dict:
    accuracy_reward = 0.0
    # math_verify
    try:
        parsed_ground_truth = parse(sol)
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
            if grade_answer(extracted_answer, sol):
                accuracy_reward = 1.0
        except Exception as e:
            print(f"mathruler failed: {e}. Extracted Answer: {extracted_answer}, Ground Truth: {sol}")
            pass
    return accuracy_reward
def judge_multi_choice(predict_str: str, ground_truth: str) -> dict:
    if ground_truth[-1] == '.':
        ground_truth = ground_truth[:-1]
    extracted_str = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    if extracted_str is not None:
        predict_str = extracted_str
    def extract_choice(answer):
        import re
        pattern = r"([A-Z])\."
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
    return accuracy_reward

def acc_reward(predict_str: str, sol: str) -> float:
    reward = 0.0
    # Try symbolic verification first
    gold_parsed = parse(sol,extraction_config=[LatexExtractionConfig(),ExprExtractionConfig()])
    if len(gold_parsed) > 0:
        try:
            predict_str_in_ans_tag = re.search(r'<answer>(.*?)</answer>', predict_str, re.DOTALL)
            if predict_str_in_ans_tag:
                predict_str_in_ans_tag = predict_str_in_ans_tag.group(1).strip()
                answer_parsed = parse(predict_str_in_ans_tag, extraction_config=[LatexExtractionConfig(boxed_match_priority=0),ExprExtractionConfig()])
                if float(verify(gold_parsed, answer_parsed)) > 0:
                    reward = 1.0
            else:
                # print(f"Failed to extract answer from: {predict_str}")
                pass
        except Exception as e:
            print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', predict_str, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else predict_str.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
    else:
        print("Failed to parse gold solution: ", sol)
    return reward

def compute_score(prompt: str, predict_str: list, ground_truth: str, extra_info: dict = None) -> float:
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
    acc = judge_multi_choice(' '.join(predict_str), ground_truth)
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
    return score, acc_score, format_score

if __name__ == '__main__':
    question = "Elena Ferrante" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = ["""<think>\nTo answer the question, I will locate Kevin Watson\'s transfer in the table and identify the fee information associated with it.\n</think>\n<video_zoom></video_zoom>""", """<think>\nI think.\n</think>\n<answer>A\n</answer>"""]
    ground_truth = "A" #"Martha White" #"china" #"$ 2 $" #"A" #"1:3" #"0.5 cm" #"0.5"
    extra_info = {
        "acc_reward_weight": 0.95,
        "format_reward_weight": 0.05,
    }
    s1 = compute_score(question, predict_str, ground_truth, extra_info)
    print(s1)

    s2 = format_reward(predict_str, extra_info)
    print(s2)
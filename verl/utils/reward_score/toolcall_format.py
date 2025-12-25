import re
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
import torch
import json

def is_valid_tool_call(response, step_tool_call_format) -> bool:
    pattern = step_tool_call_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <video_zoom> Count
    if response.count('<video_zoom>') != 1 or response.count('</video_zoom>')!= 1:
        return False
    # 3). <answer> or </answer> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    return True
def strict_is_valid_tool_call(response, step_tool_call_format, max_frame_num = 16) -> bool:
    is_in_frame_limit = False
    try:
        pattern = re.compile(r'<video_zoom>(.*?)</video_zoom>', re.DOTALL)
        tool_call_content = pattern.findall(response)
        if tool_call_content is None:
            return False
        tool_call_content = json.loads(tool_call_content[-1])
        tool_call_frame_count = int(float(tool_call_content['fps']) * (float(tool_call_content['segment'][1]) - float(tool_call_content['segment'][0])))
        # is_in_frame_limit = (float(tool_call_content['fps']) * (float(tool_call_content['segment'][1]) - float(tool_call_content['segment'][0])) <= max_frame_num)
        is_in_frame_limit = (tool_call_frame_count <= max_frame_num)
        return (is_in_frame_limit and is_valid_tool_call(response, step_tool_call_format)), tool_call_frame_count
    except Exception as e:
        print(f"is_valid_tool_call failed: {e}")
        return False, 0
    pass
def is_valid_direct_answer(response, direct_answer_format) -> bool:
    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). <video_zoom> is not allowed!
    if '<video_zoom>' in response:
        return False
    return True
def format_reward(predict_str_list: list, extra_info: dict = None):
    conv_rounds = len(predict_str_list)
    format_score, tool_call_count = 0, 0
    # All allowed formats
    direct_answer_format = r'^<think>.*</think>.*<answer>.*</answer>$'
    step_tool_call_format = r'^<think>.*</think>.*<video_zoom>.*</video_zoom>'
    tool_match = r'<video_zoom>.*</video_zoom>'
    tool_call_frame_num = 0
    # HACK/FIXME: We need more flexible judge in the future
    # 1-turn
    if conv_rounds == 1:
        response = predict_str_list[0].strip()
        if re.search(tool_match, response, re.DOTALL):
            tool_call_count += 1
        # Direct Answer
        if is_valid_direct_answer(response, direct_answer_format):
            format_score = 1
    # multi-turn
    else:
        tool_call_match_flag = True
        for response in predict_str_list[:-1]:
            response = response.strip()
            if re.search(tool_match, response, re.DOTALL):
                tool_call_count += 1
            # Call Function Tool
            result = strict_is_valid_tool_call(response, step_tool_call_format, extra_info['tool_call_max_frames'])
            if not result[0]:
                tool_call_match_flag = False
                break
            tool_call_frame_num += result[1]
        final_answer_match_flag = is_valid_direct_answer(predict_str_list[-1], direct_answer_format)
        if tool_call_match_flag and final_answer_match_flag:
            format_score = 1
    return format_score, tool_call_count, tool_call_frame_num

if __name__ == "__main__":
    predict_str_list = [
        "<think>Let's think step by step.</think>\n<answer><video_zoom>{\"fps\": 16, \"segment\": [0, 1]}</video_zoom>",
        "<think></think><answer>0</answer>",
    ]
    print(format_reward(predict_str_list))

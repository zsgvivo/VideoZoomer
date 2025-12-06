def tool_call_extra_score(tool_call_count,tool_call_frame_num, acc, extra_info: dict = None) -> float:
    tool_call_penalty = extra_info['tool_call_penalty']
    penalize_correct_tool_call = extra_info['penalize_correct_tool_call']
    penalize_incorrect_tool_call = extra_info['penalize_incorrect_tool_call']
    penalize_all_tool_call = extra_info['penalize_correct_tool_call'] and extra_info['penalize_incorrect_tool_call']
    successful_tool_call_rate = extra_info.get('success_tool_call_rate', 0.0)
    target_tool_call_rate = extra_info.get('target_tool_call_rate', 1.0)
    penalty_per_turn = extra_info.get('penalty_per_turn', 0.0)
    # set target tool call frame number
    tool_call_frame_num_target = extra_info.get('tool_call_frame_target', 0.0) + 1e-4
    tool_call_frame_num_penalty = extra_info.get('tool_call_frame_num_penalty', 0.0)
    extra_score = 0
    if penalize_all_tool_call:
        if target_tool_call_rate > 0 and successful_tool_call_rate <= target_tool_call_rate and tool_call_count > 0:
            extra_score -= tool_call_penalty
            extra_score -= penalty_per_turn * tool_call_count
            extra_score -= tool_call_frame_num_penalty if tool_call_frame_num >= tool_call_frame_num_target else 0
    elif penalize_correct_tool_call:
        if acc == 1 and tool_call_count > 0 and successful_tool_call_rate <= target_tool_call_rate:
            extra_score -= tool_call_penalty
            extra_score -= penalty_per_turn * tool_call_count
            extra_score -= tool_call_frame_num_penalty if tool_call_frame_num >= tool_call_frame_num_target else 0
    elif penalize_incorrect_tool_call:
        if acc == 0 and tool_call_count > 0 and successful_tool_call_rate <= target_tool_call_rate:
            extra_score -= tool_call_penalty
            extra_score -= penalty_per_turn * tool_call_count
            extra_score -= tool_call_frame_num_penalty if tool_call_frame_num >= tool_call_frame_num_target else 0
    return extra_score

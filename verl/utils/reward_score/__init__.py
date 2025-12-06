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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['SynthLabsAI/Big-Math-RL-Verified', 'DigitalLearningGmbH/MATH-lighteval', 'HuggingFaceH4/aime_2024', 'AI-MO/aimo-validation-amc', 'HuggingFaceH4/MATH-500']:
        # from . import openr1
        # res = openr1.compute_score(solution_str, ground_truth)
        raise NotImplementedError
    elif data_source in ['nextqa', 'LLaVA-Video-178K', 'MVLU', 'youtube', 'lvb', 'mvbench', 'VideoMME', 'tempcompass', 'mmvu', 'lvbench', 'longvideo-reason','MVLU_test', 'glimpse', 'LSDBench', 'CLEVRER_validation']:
        if extra_info['answer_type'] == 'multiple_choice':
            if extra_info['judge_mc_by_gpt']:
                from . import general_qa
                res = general_qa.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
            else:
                from . import openr1
                res = openr1.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        elif extra_info['answer_type'] == 'open_ended':
            from . import general_qa
            res = general_qa.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        else:
            raise NotImplementedError
        # from . import general_qa_v6
        # res = general_qa_v6.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    elif 'Video-MMLU' in data_source:
        from . import video_mmlu
        res = video_mmlu.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
    else:
        print(f'data_source {data_source} unregistered, use gpt judge')
        from . import general_qa
        res = general_qa.compute_score(kwargs['prompt'], solution_str, ground_truth, extra_info)
        # raise NotImplementedError

    # if isinstance(res, (int, float, bool)):
    #     return float(res)
    # else:
    #     return float(res[0])
    return res

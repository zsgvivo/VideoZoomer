import re
import ast
import openai
from retrying import retry

client = openai.AzureOpenAI(
    azure_endpoint="https://search-va.byteintl.net/gpt/openapi/online/multimodal/crawl",
    api_version="2023-07-01-preview",
    api_key="54nhP5uBXv7iWgHJ4bWMD90Nwkn09BXN"
)
DAFAULT_SYSTEM_PROMPT=("You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the correctness of the prediction compared to the answer.")
DEFAULT_USER_PROMPT=("I will give you a question related to an image and the following text as inputs:\n\n"
                    "1. **Question Related to the Image**: {question}\n"
                    "2. **Ground Truth Answer**: {ground_truth}\n"
                    "3. **Model Predicted Answer**: {prediction}\n\n"
                    "Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:"
                    "- **Relevance**: Does the predicted answer directly address the question posed. considering the information provided by the given question?"
                    "- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:"
                    "(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct."
                    "(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct."
                    "**Output Format**:"
                    "Your response should include an explanation of your judgement and an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model's prediction strictly aligns with the ground truth, while 0 means it does not."
                    "Explanation: <brief judgement of prediction>"
                    "Score: <0-1>")
MV_MATH_MUTICHOICE_SYSTEM_PROMPT = """
        You are a math expert. Given a question, a model's answer, and the standard answer, determine if the model's answer is correct. Only Respond with 'true' if the model's answer is correct, otherwise respond with 'false', without any additional information.
        For example:  
        Question: You are a expert in math problem solving.Structure Clauses: line Q R, line S T R, line Q P S, line P T\nSemantic Clauses: PT \\parallel QR\nIf ST = 8, TR = 4, and PT = 6, find QR. Choices are A: 6.0, B: 8.0, C: 9.0, D: 10.0. 
        Model Answer: The answer is C
        Standard Answer: 9.000
        Is the model's answer correct? true
        
        Question: <image>\nYou are a expert in math problem solving.Structure Clauses: line J I, line G H I, line G K J, line K H\nSemantic Clauses: KH \\parallel JI\nFind GI if GH = 9, GK = 6, and KJ = 4. Choices are A: 6.0, B: 9.0, C: 12.0, D: 15.0.
        Model Answer:Since KH is parallel to JI, we have angle GKJ = angle HGI. Also, since GH = 9, GK = 6, and KJ = 4, we can calculate that GH/GK = KJ/JI. Therefore, 9/6 = 4/JI. Solving for JI, we find that JI = 8. Thus, GI = GH - JI = 9 - 8 = 1. Therefore, the answer is D.\nAnswer:D
        Standard Answer: 15.000
        Is the model's answer correct? true
        
        Question: <image>\nYou are a expert in math problem solving.Structure Clauses: line P T, line S Q T, line S R, line P Q R\nSemantic Clauses: SR \\parallel PT, SQ = 3+x, TQ = 3, PQ = 6-x, RQ = 6+x\nFind PQ. Choices are A: 3.0, B: 4.0, C: 6.0, D: 9.0.
        Model Answer: The answer is B
        Standard Answer: 6.000
        Is the model's answer correct? false
    """
MV_MATH_MUTICHOICE_USER_PROMPT="""
        Question: {question}
        Model Answer: {ground_truth}
        Standard Answer: {prediction}
        Is the model's answer correct? Just respond with 'true' or 'false'.
    """
MV_MATH_SINGLE_STEP_SYSTEM_PROMPT = """You are a math expert. You need to check if the model's final answer matches the standard answer."""
MV_MATH_SINGLE_STEP_USER_PROMPT = """
    Standard answer: {ground_truth}
    Model's response: {prediction}
    Do they match? Just answer 'true' if correct, 'false' if incorrect.
    """
MV_MATH_MULTI_STEP_SYSTEM_PROMPT = """
You are a math expert. You need to check whether the model's final answer is consistent with the standard answer. Your answer is in the form of correct step/total step. For example, if there are three questions in the question, and the model correctly answers 2 questions, then the output is 2/3. If the model correctly answers 0 questions, then the output is 0/3.
"""
MV_MATH_MULTI_STEP_USER_PROMPT = """
    Standard answer: {ground_truth}
    Model's response: {prediction}
    Only output correct step/total step, without any other steps.
    """

@retry(wait_exponential_multiplier=200, wait_exponential_max=2000, retry_on_exception=lambda e: isinstance(e, openai.RateLimitError))
def match_by_gpt4o(question, ground_truth, prediction, prompt_type="default"):
    if prompt_type == "default":
        system_prompt=DAFAULT_SYSTEM_PROMPT
        user_prompt=DEFAULT_USER_PROMPT.format(question=question, ground_truth=ground_truth, prediction=prediction)
    elif prompt_type == "mvmath_choice" or 'Multiple_Choices' in prompt_type or 'Single_Choice' in prompt_type:
        system_prompt = MV_MATH_MUTICHOICE_SYSTEM_PROMPT
        user_prompt = MV_MATH_MUTICHOICE_USER_PROMPT.format(question=question, ground_truth=ground_truth, prediction=prediction)
    elif prompt_type == "mvmath_single-step" or 'Open-ended' in prompt_type or "OlympiadBench" in prompt_type:
        system_prompt = MV_MATH_SINGLE_STEP_SYSTEM_PROMPT
        user_prompt = MV_MATH_SINGLE_STEP_USER_PROMPT.format(ground_truth=ground_truth, prediction=prediction)
    elif prompt_type == "mvmath_multi-step" or 'Open_Ended' in prompt_type:
        system_prompt = MV_MATH_MULTI_STEP_SYSTEM_PROMPT
        user_prompt = MV_MATH_MULTI_STEP_USER_PROMPT.format(ground_truth=ground_truth, prediction=prediction)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    @retry(stop_max_attempt_number=5, wait_exponential_multiplier=200)
    def query_gpt4o(question, ground_truth, prediction):
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            temperature=0,
            max_tokens=512,
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            timeout=120,
        )
        # Convert response to a Python dictionary.
        response_message = completion.choices[0].message.content
        print(f"Response Message: {response_message}")
        if prompt_type == "default":
            # 正则表达式匹配 'Score: ' 后面的数字
            score_match = re.search(r'Score:\s*(\d+/\d+|\d+)', response_message)
            if score_match:
                score = score_match.group(1)
                # 如果是分数形式，转换为 float
                if '/' in score:
                    numerator, denominator = map(int, score.split('/'))
                    return numerator / denominator
                # 如果是整数，转换为整数类型
                else:
                    return float(score)
        elif prompt_type == "mvmath_choice" or 'Multiple_Choices' in prompt_type or 'Single_Choice' in prompt_type:
            # 正则表达式匹配 'true'或'false'
            score_match = re.search(r'(true|false)', response_message, re.IGNORECASE)
            if score_match:
                score = score_match.group(1)
                # 如果是分数形式，转换为 float
                if score.lower() == 'true':
                    return 1.0
        elif prompt_type == "mvmath_single-step" or 'Open-ended' in prompt_type or "OlympiadBench" in prompt_type:
            # 正则表达式匹配 'true'或'false'
            score_match = re.search(r'(true|false)', response_message, re.IGNORECASE)
            if score_match:
                score = score_match.group(1)
                # 如果是分数形式，转换为 float
                if score.lower() == 'true':
                    return 1.0
        elif prompt_type == "mvmath_multi-step" or 'Open_Ended' in prompt_type:
            score = response_message
            # 如果是分数形式，转换为 float
            if '/' in score:
                numerator, denominator = map(int, score.split('/'))
                return numerator / denominator
            # 如果是整数，转换为整数类型
            else:
                return float(score)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

    return query_gpt4o(question, ground_truth, prediction)

if __name__ == "__main__":
    question = "How long does it take to arrive at the hotel?"
    ground_truth = "75min"
    prediction = "1.25h"
    print("Score: ", match_by_gpt4o(question, ground_truth, prediction))

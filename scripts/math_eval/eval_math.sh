pip install git+https://github.com/huggingface/lighteval.git@ed084813e0bd12d82a06d9f913291fdbee774905
pip3 install lighteval[math]
pip3 install more_itertools

cd /opt/tiger/aiic_verl/scripts/math_eval;
mkdir /opt/tiger/checkpoints;
hdfs dfs -get hdfs://harunava/home/byte_data_seed_azureb_tteng/user/lijunyi/trials/verl_qwen25vl_grpo_mm_eureka/verl_qwen25_vl_7B_grpo_MM_Eureka_bs1024_n8_mini256_micro16_KL1_3_img-text_general_reason_prompt_5epochs/global_step60 /opt/tiger/checkpoints/;

NUM_GPUS=4
MODEL=/opt/tiger/checkpoints/global_step60
BASE_NAME=$(basename "$MODEL")
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:2048,temperature:0.0,top_p:1.0}"
OUTPUT_DIR=/opt/tiger/aiic_verl/logs/$BASE_NAME

export VLLM_WORKER_MULTIPROC_METHOD=spawn

lighteval vllm $MODEL_ARGS "custom|minervamath|0|0,custom|olympiadbench|0|0,custom|amc|0|0,custom|aime24|0|0,custom|math_500|0|0" \
    --custom-tasks /opt/tiger/aiic_verl/scripts/math_eval/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR


# # GPQA Diamond
# TASK=gpqa:diamond
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks /opt/tiger/aiic_verl/scripts/math_eval/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# # LiveCodeBench
# lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 
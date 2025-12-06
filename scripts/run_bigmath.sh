set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=['/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/Big-Math-RL-Verified/Big-Math-RL-Verified.parquet'] \
    data.val_files=['/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/Big-Math-RL-Verified/Big-Math-RL-Verified-val.parquet,/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/aime_2024/aime_2024.parquet,/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/aimo-validation-amc/aimo-validation-amc.parquet,/mnt/bn/tiktok-mm-4/aiic/users/dingyang/data/MATH-500/MATH-500.parquet'] \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.image_key=images \
    'data.post_prompt="\nOutput the thinking process within <think> </think> tags and final answer within <answer> </answer> tags. The final answer should contain \\boxed{{}}."' \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1e-4 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_bigmath_debug' \
    trainer.experiment_name='qwen_7b_function_rm' \
    trainer.log_training_rollouts_freq=5 \
    trainer.train_generations_to_log_to_wandb=256 \
    trainer.val_generations_to_log_to_wandb=128 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    reward_model.log_rewards_separately=True \
    trainer.reflection_keywords=['wait,recheck,alternatively,retry,however'] 
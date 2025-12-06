#!/usr/bin/env bash
set -euo pipefail
set -x

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1

DATA_DIR=/workspace/data/gsm8k
REWARD_FN_PATH=/workspace/rl_cot_monitorability/scripts/gsm8k_reward.py
REWARD_FN_NAME=compute_score
ACTOR=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
CRITIC=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
PROJECT=verl_gsm8k_ppo_DeepSeek-R1-Distill-Qwen-7B
EXP=DeepSeek-R1-Distill-Qwen-7B_kl-0.05+klInRewardTrue_$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=/workspace/results/${EXP}

# Create results directory
mkdir -p ${RESULTS_DIR}

python3 -m verl.trainer.main_ppo \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_penalty=low_var_kl \
 algorithm.kl_ctrl.kl_coef=-0.05 \
 algorithm.kl_ctrl.type="fixed" \
 data.train_files=${DATA_DIR}/train.parquet \
 data.val_files=${DATA_DIR}/test.parquet \
 data.train_batch_size=1024 \
 data.max_prompt_length=180 \
 data.max_response_length=1024 \
 data.filter_overlong_prompts=True \
 data.truncation=error \
 actor_rollout_ref.model.path=${ACTOR} \
 actor_rollout_ref.model.lora_rank=16 \
 actor_rollout_ref.model.lora_alpha=32 \
 actor_rollout_ref.model.target_modules="all-linear" \
 actor_rollout_ref.actor.optim.lr=5e-5 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=-0.05 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.strategy=fsdp2 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.temperature=1.0 \
 actor_rollout_ref.rollout.top_p=0.95 \
 actor_rollout_ref.rollout.n=5 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
 actor_rollout_ref.rollout.load_format="safetensors" \
 actor_rollout_ref.ref.fsdp_config.param_offload=False \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
 actor_rollout_ref.ref.strategy=fsdp2 \
 custom_reward_function.path=${REWARD_FN_PATH} \
 custom_reward_function.name=${REWARD_FN_NAME} \
 critic.model.path=${CRITIC} \
 critic.optim.lr=5e-5 \
 critic.ppo_micro_batch_size_per_gpu=8 \
 critic.ppo_mini_batch_size=64 \
 critic.model.enable_gradient_checkpointing=True \
 critic.model.fsdp_config.param_offload=False \
 trainer.critic_warmup=0 \
 trainer.logger='["console","wandb"]' \
 trainer.project_name=${PROJECT} \
 trainer.log_val_generations=15 \
 trainer.experiment_name=${EXP} \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=3 \
 trainer.total_epochs=1 \
 2>&1 | tee ${RESULTS_DIR}/training.log

echo "Training complete! Logs saved to ${RESULTS_DIR}"
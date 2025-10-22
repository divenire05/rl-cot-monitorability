#!/usr/bin/env bash
set -euo pipefail
set -x

DATA_DIR=/workspace/data/gsm8k
ACTOR=Qwen/Qwen2.5-1.5B-Instruct
CRITIC=Qwen/Qwen2.5-1.5B-Instruct
PROJECT=verl_gsm8k_ppo_qwen1.5B
EXP=qwen1.5b_kl0.05_$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=/workspace/results/${EXP}

# Create results directory
mkdir -p ${RESULTS_DIR}

python3 -m verl.trainer.main_ppo \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/test.parquet \
  data.train_batch_size=2048 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=${ACTOR} \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.val_kwargs="{do_sample:false}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  critic.model.path=${CRITIC} \
  critic.model.lora_rank=16 \
  critic.model.lora_alpha=32 \
  critic.optim.lr=1e-4 \
  critic.ppo_micro_batch_size_per_gpu=8 \
  critic.ppo_mini_batch_size=256 \
  critic.model.enable_gradient_checkpointing=False \
  critic.model.fsdp_config.param_offload=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.05 \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=${PROJECT} \
  trainer.experiment_name=${EXP} \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  trainer.total_epochs=10 \
  2>&1 | tee ${RESULTS_DIR}/training.log

echo "Training complete! Logs saved to ${RESULTS_DIR}"
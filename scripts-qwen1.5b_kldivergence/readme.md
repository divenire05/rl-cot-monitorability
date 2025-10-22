playing around with different kl divergence values for this set of experiments
tried: 0.001, 0.01, 0.05
left to try: 0.2

Didn't plot metrics for the 3 runs yet

run eval script command: python eval_monitorability.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --lora-path /workspace/rl_cot_monitorability/scripts/run1/checkpoints/verl_gsm8k_ppo_qwen1.5B/qwen1.5b_kl0.001_20251021_175225/global_step_30/actor/lora_adapter \
  --data /workspace/data/gsm8k/test.parquet \
  --out /workspace/rl_cot_monitorability/scripts/run1/eval_results.jsonl \
  --temperature 0.7

trainer.save_freq=10 \    # play around with this line based on amount of disk space available and how detailed? you want the results graphs to be

trainer.n_gpus_per_node=2 \   # depending on how many GPUs you are using

data.max_response_length=1024 \   # maybe try 2048 if its getting cutoff here?
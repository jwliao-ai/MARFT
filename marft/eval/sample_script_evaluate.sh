#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python evaluate.py \
  --evaluator_type math \
  --model_path model_path \
  --data_path data_path \
  --profile_path ../scripts/profiles/math_dual.json \
  --lora_path lora_checkpoint_path \
  --output_dir output_dir_name \
  --response_filename responses.json \
  --metrics_filename metrics.json \
  --metrics_timestamp \
  --num_agents num_agents \
  --context_window 2048 \
  --top_k 50 \
  --top_p 0.95 \
  --temperature 0.5 \
  --max_new_tokens 1024 \
  --do_sample \
export CUDA_VISIBLE_DEVICES=1
echo $CUDA_VISIBLE_DEVICES
python train_math.py \
        --seed 10 \
        --env_name math_env \
        --algorithm_name APPO \
        --experiment_name experiment_name \
        --dataset_name MATH \
        --flag train \
        --num_mini_batch 1 \
        --ppo_epoch 1 \
        --lr 1e-6 \
        --critic_lr 5e-5 \
        --dataset_path ../envs/math/benchmarks/MATH/train.json \
        --model_name_or_path path_to_the_model \
        --n_agents 2 \
        --agent_iteration_interval 1000 \
        --profile_path profiles/math_dual.json \
        --n_rollout_threads 4 \
        --episode_length 1 \
        --gradient_cp_steps 2 \
        --context_window 2048 \
        --max_new_tokens 512 \
        --save_interval 1000
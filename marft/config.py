import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["APPO", "TPPO"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)

    Env parameters:
        --env_name <str>
            specify the name of environment
        --dataset_name <str>
            specify the name of dataset
        --dataset_path <str>
            specify the path of dataset
        --flag <str>
            specify the flag of dataset, including `["train", "test"]`

    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer.

    Network parameters:
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.

    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)

    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.
        --huber_delta <float>
            coefficient of huber loss.

    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate

    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.

    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.

    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.

    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """
    parser = argparse.ArgumentParser(description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="APPO", choices=["TPPO", "APPO"])
    parser.add_argument("--normalization_mode", type=str, default="sum", choices=["token", "word", "sum"], help="specify the normalization mode for APPO (use TWOSOME or not).")
    parser.add_argument("--agent_iteration_interval", type=int, default=0, help="the interval of agent iteration. if set 0, train concurrently.")
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.",)
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch.")
    parser.add_argument("--cuda", action="store_false", default=True, help="by default True, will use GPU to train; or else will use CPU.")
    parser.add_argument("--cuda_deterministic", action="store_false", default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int, default=1, help="Number of torch threads for training.")
    parser.add_argument("--n_rollout_threads", type=int, default=32, help="Number of parallel envs for training rollouts.")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1, help="Number of parallel envs for evaluating rollouts.")
    parser.add_argument("--num_env_steps", type=int, default=10e6, help="Number of environment steps to train (default: 10e6).")
    parser.add_argument("--horizon", type=int, default=3, help="The horizon of the rollout.")

    # env parameters
    parser.add_argument("--env_name", type=str, default="MATH", help="Which env to run on")
    parser.add_argument("--dataset_name", type=str, default="xlam", help="Which dataset to test on")
    parser.add_argument("--dataset_path", type=str, help="path to dataset")
    parser.add_argument("--flag", type=str, default="train", help="flag to distinguish different runs")

    # mas parameters
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Which base model to use")
    parser.add_argument("--load_path", type=str, default=None, help="path to the checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="max_new_tokens")
    parser.add_argument("--n_agents", type=int, default=1)
    parser.add_argument("--profile_path", type=str, default="agent_profiles.json", required=True)
    parser.add_argument("--context_window", type=int, default=2048, help="the context window of the actor when acting")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--warmup_steps", type=int, default=0, help="number of warmup steps for the critic")
    parser.add_argument("--hidden_size", type=int, default=64, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_orthogonal", action="store_false", default=True, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-7, help="learning rate (default: 5e-7)")
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="critic learning rate (default: 5e-4)")
    parser.add_argument("--opti_eps", type=float, default=1e-5, help="RMSprop optimizer epsilon (default: 1e-5)")
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--gradient_cp_steps", type=int, default=1, help="number of gradient accumulation steps (default: 1)")
    parser.add_argument("--ppo_epoch", type=int, default=5, help="number of ppo epochs (default: 5)")
    parser.add_argument("--use_clipped_value_loss", action="store_false", default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2, help="ppo clip parameter (default: 0.2)")
    parser.add_argument("--num_mini_batch", type=int, default=4, help="number of batches for ppo (default: 1)")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value_loss_coef", type=float, default=1, help="value loss coefficient (default: 0.5)")
    parser.add_argument("--use_max_grad_norm", action="store_false", default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max norm of gradients (default: 0.5)")
    parser.add_argument("--use_gae", action="store_false", default=True, help="use generalized advantage estimation")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards (default: 0.99)")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="gae lambda parameter (default: 0.95)")
    parser.add_argument("--use_proper_time_limits", action="store_true", default=False, help="compute returns taking into account time limits")
    parser.add_argument("--use_huber_loss", action="store_false", default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks", action="store_false", default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks", action="store_false", default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help="coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action="store_true", default=False, help="use a linear schedule on the learning rate")
    
    # save parameters
    parser.add_argument("--save_interval", type=int, default=100, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=1, help="time duration between contiunous twice log printing.",)

    # eval parameters
    parser.add_argument("--use_eval", action="store_true", default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=5, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=10, help="number of episodes of a single evaluation.")

    return parser

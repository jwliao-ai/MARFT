#!/usr/bin/env python
import sys
import os
import numpy as np
from pathlib import Path
import torch
import yaml

sys.path.append("../../")
from marft.config import get_config
from marft.envs.math.math_env import MathEnv
from marft.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from marft.runner.shared.math_runner import MathRunner as Runner


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MathEnv(
                rank=rank,
                model_name=all_args.base_model,
                num_agents=all_args.n_agents,
                profile_path=all_args.profile_path,
                dataset_path=all_args.dataset_path,
                horizon=all_args.horizon,
                mode="train",
            )
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return ShareDummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MathEnv(
                rank=rank,
                model_name=all_args.base_model,
                num_agents=all_args.n_agents,
                profile_path=all_args.profile_path,
                dataset_path=all_args.dataset_path,
                horizon=all_args.horizon,
                mode="test",
            )
            env.seed(all_args.seed + rank * 5000)
            return env
        return init_env
    return ShareDummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    all_args.base_model = Path(all_args.model_name_or_path).parts[-1]
    return all_args

def save_args_to_yaml(args, filename="args.yaml"):
    """Save argparse arguments to a YAML file."""
    with open(filename, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)

def build_run_dir(all_args):
    run_dir = (
        Path(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
            + "/scripts/results"
        )
        / all_args.experiment_name
        / all_args.base_model
        / all_args.dataset_name
        / all_args.algorithm_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        curr_run = "run_1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("_")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run_1"
        else:
            curr_run = "run_%i" % (max(exst_run_nums) + 1)
    curr_run += f"_agent#{all_args.n_agents}_seed{all_args.seed}"
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    print(f"Saving results to {run_dir}")
    return run_dir

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    run_dir = build_run_dir(all_args)
    save_args_to_yaml(all_args, run_dir / "args.yaml")

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    # eval_envs = make_eval_env(all_args)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": envs.n_agents if envs is not None else 1,
        "run_dir": run_dir,
    }

    runner = Runner(config)
    runner.run()

    # post process
    if envs is not None:
        envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

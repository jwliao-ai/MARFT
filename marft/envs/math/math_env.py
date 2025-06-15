import numpy as np
import json
import random
from typing import Optional
from .parse_utils_qwen import extract_answer as extract_fn, parse_ground_truth
from .grader import math_equal

# training data with mode="train" and testing data with mode="test"
def load_dataset(dataset_path, mode):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    return dataset

def load_profiles(path):
    with open(path, 'r') as file:
        profiles = json.load(file)
    return profiles

def extract_answer(answer_str: str) -> str:
    return extract_fn(answer_str, data_name='math')

def extract_groundtruth(groundtruth_str: str) -> str:
    return parse_ground_truth(groundtruth_str, data_name='math')

def judge_correct(extracted_groundtruth: Optional[str], answer: str) -> bool:
    result = math_equal(answer, extracted_groundtruth)
    return result

class MathEnv:

    def __init__(self, rank, model_name, num_agents, profile_path, dataset_path, horizon, mode):
        
        self.rank = rank
        self.mode = mode
        self.model_name = model_name
        self.dataset = load_dataset(dataset_path=dataset_path, mode=mode)
        self.profiles = load_profiles(profile_path)
        self.n_agents = num_agents
        assert self.n_agents == len(self.profiles), "Number of agents must match the number of profiles."
        self.max_steps = horizon
        self.step_count = 0
        
        self.problem = None
        self.label = None
        self.current_state = None

    def reset(self):
        problem_answer_pair = random.choice(self.dataset)
        self.problem = problem_answer_pair["problem"]
        self.label = problem_answer_pair["final_answer"]
        self.current_state = '<|im_start|>problem: ' + self.problem + "<|im_end|>\n"
        self.history = []
        obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        self.step_count = 0
        return obs
    
    def step(self, actions):
        self.step_count += 1
        actions_to_check = []
        self.state_transition(actions)

        for i in range(self.n_agents):
            if self.profiles[i]["with_answer"]:
                actions_to_check.append(actions[i])

        score = 0.0
        for action in actions_to_check:
            if self._is_correct(action): 
                score += 1.0
        score /= len(actions_to_check) # normalize
        
        if score > 0.0 or self.step_count >= self.max_steps:
            dones = np.ones((self.n_agents), dtype=bool)
            # score -= self.step_count # penalize for more steps
        else:
            dones = np.zeros((self.n_agents), dtype=bool)
            
        if score == 0.0:
            self.current_state = self.current_state + "judge: The answer is incorrect.\n"
        else:
            self.current_state = self.current_state + "judge: The answer is correct.\n"

        next_obs = np.array([self.current_state for _ in range(self.n_agents)], dtype=np.object_)
        rewards = [0 if idx != self.n_agents - 1 else score for idx in range(self.n_agents)]
        infos = {"state": self.current_state, "gt": self.label, "episodic_return": score}
        return next_obs, rewards, dones, infos

    def state_transition(self, actions):
        for i, action in enumerate(actions):
            self.current_state = self.current_state + self.profiles[i]["role"] + ": " + action + "\n"

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        return judge_correct(self.label, extracted_answer)

    def seed(self, seed):
        np.random.seed(seed)

    def get_env_info(self):
        env_info = {"n_agents": self.n_agents}
        return env_info
    
    def close(self):
        pass 
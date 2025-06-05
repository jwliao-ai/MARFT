from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from marft.inference.inference_engine import InferenceEngine
import torch
import os
import json
import numpy as np
from abc import ABC
from torch.distributions.categorical import Categorical
from peft import (
    PeftModel,
    LoraConfig,
    PeftConfig,
    get_peft_model,
)

def load_profiles(path):
    with open(path, 'r') as file:
        profiles = json.load(file)
    return profiles

class MAS(ABC):

    def __init__(
            self, 
            model_path: str | os.PathLike, 
            context_window: int, 
            max_new_tokens: int, 
            num_agents: int, 
            profile_path: str | os.PathLike,
            algo: str = "APPO", 
            normalization_mode: str = "sum",
            load_path: str = None,
            load_in_4bit: bool = False,
            bf16: bool = True,
            device_map = None,
            use_vllm: bool = False,
            multi_gpu: bool = False,
            **kwargs,
        ):
        self.device = "cuda:0"
        self.algo = algo
        self.normalization_mode = normalization_mode
        self.num_agents = num_agents
        if load_in_4bit:
            assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            nf4_config = None
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else 'auto',
            quantization_config=nf4_config, 
            device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=False, 
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.profiles = load_profiles(profile_path)
        self.use_vllm = use_vllm
        self.multi_gpu = multi_gpu

        # inference engine for trajectory collection
        tensor_parallel = torch.cuda.device_count() if multi_gpu else 1
        self.inference_engine = InferenceEngine(model_path, max_new_tokens, use_vllm=use_vllm, tensor_parallel_size=tensor_parallel)

        self.agents = self._init_agents(load_path).to(self.device)
        if self.multi_gpu and torch.cuda.device_count() > 1:
            self.agents = torch.nn.DataParallel(self.agents)
        self.critic = self._init_critic(load_path).to(self.device)
        if self.multi_gpu and torch.cuda.device_count() > 1:
            self.critic = torch.nn.DataParallel(self.critic)

    @property
    def agent_model(self):
        return self.agents.module if isinstance(self.agents, torch.nn.DataParallel) else self.agents

    def _init_agents(self, lora_path=None):
        self.base_model.enable_input_require_grads()
        model = None
        if lora_path is None:
            # Initialize all adapters from scratch
            for i in range(self.num_agents):
                config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                if model is None:
                    model = get_peft_model(self.base_model, config, adapter_name=self.profiles[i]["role"])
                else:
                    model.add_adapter(adapter_name=self.profiles[i]["role"], peft_config=config)
            model.print_trainable_parameters()
            print(f"lora_path is None, initialize {self.num_agents} adapters from scratch")
        else:
            if not os.path.exists(lora_path):
                raise ValueError(f"LoRA path does not exist: {lora_path}")
            adapter_dirs = [d for d in os.listdir(lora_path) if os.path.isdir(os.path.join(lora_path, d))]
            if len(adapter_dirs) != self.num_agents:
                raise ValueError(f"Number of pretrained weights ({len(adapter_dirs)}) must match num_agent ({self.num_agents})")
            for adapter_name in adapter_dirs:
                adapter_path = os.path.join(lora_path, adapter_name)
                if model is None:
                    model = PeftModel.from_pretrained(self.base_model, adapter_path, adapter_name=adapter_name)
                else:
                    model.add_adapter(adapter_name=adapter_name, peft_config=PeftConfig.from_pretrained(adapter_path))
                print(f"lora_path is {lora_path}, load {self.num_agents} adapters from pretrained weights")
        # Apply half-precision across all adapters
        model.half()
        print(f"Initialized model with {self.num_agents} adapters.")
        return model

    def _init_critic(self, critic_path=None):
        if self.algo == "APPO":
            from marft.critics import ActionCritic
            critic = ActionCritic(self.base_model, self.tokenizer)
        elif self.algo == "TPPO":
            from marft.critics import TokenCritic
            critic = TokenCritic(self.base_model, self.tokenizer)
        else:
            raise NotImplementedError
        if critic_path is not None:
            critic_path = os.path.join(critic_path, "value_head.pth")
            critic.load_value_head(critic_path, map_location="cpu")
            print(f"Load critic from {critic_path}")
        return critic

    @torch.no_grad()
    def get_actions_sequential(self, obs: np.ndarray):
        """
        Args:
            obs: np.ndarray of shape (rollout_threads, num_agents)

        Returns:
            all_actions: np.ndarray of shape (rollout_threads, num_agents)
            all_action_tokens: torch.tensor of shape (rollout_threads, num_agents, max_new_tokens)

        Compute actions and value function predictions for the given inputs.
        Sequentially appends responses from previous agents in the prompt.
        """
        # Note: for online, batch_size = 1, so the first dimension goes away
        rollout_threads, num_agents = obs.shape
        all_obs = np.empty((rollout_threads, num_agents), dtype=object)
        all_actions = np.empty((rollout_threads, num_agents), dtype=object)
        all_action_tokens = torch.ones((rollout_threads, num_agents, self.max_new_tokens), dtype=torch.int64, device=self.device,) * self.tokenizer.pad_token_id

        prompts = obs[:, 0].tolist()
        for agent_idx in range(num_agents):
            prompts = [prompt + "<|im_start|>" + self.profiles[agent_idx]["role"] + ": " for prompt in prompts]
            prompts_with_profile = [self.profiles[agent_idx]["prompt"] + prompt for prompt in prompts]
            token_seq = self.tokenizer(prompts_with_profile, return_tensors="pt", padding=True)
            input_ids = token_seq["input_ids"].cuda()
            attn_mask = token_seq["attention_mask"].cuda()
            if self.use_vllm:
                sequences_text = self.inference_engine.generate(prompts_with_profile)
                sequences = []
                for idx, text in enumerate(sequences_text):
                    gen_ids = self.tokenizer(text, return_tensors="pt").input_ids[0].cuda()
                    seq = torch.cat([input_ids[idx], gen_ids])
                    sequences.append(seq)
                sequences = torch.stack(sequences)
            else:
                self.agent_model.set_adapter(self.profiles[agent_idx]["role"])
                output = self.agents.generate(
                    input_ids,
                    attention_mask=attn_mask,
                    do_sample=True,
                    top_k=50,
                    temperature=0.5,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
                sequences = output.sequences
            actions = []
            for i in range(rollout_threads):
                action_token = sequences[i][input_ids[i].shape[0] :]
                all_action_tokens[i, agent_idx, : action_token.shape[0]] = action_token.clone()
                action = self.tokenizer.decode(action_token, skip_special_tokens=True)
                prompts[i] = prompts[i] + action + "<|im_end|>\n"
                actions.append(action)
            actions = np.array(actions, dtype=np.object_)
            all_obs[:, agent_idx] = np.array(prompts_with_profile, dtype=np.object_)
            all_actions[:, agent_idx] = actions
            # print(f"agent {agent_idx} actions: {actions}")
            # print(f"-" * 20)

        return all_obs, all_actions, all_action_tokens


    def get_slice(self, logits: torch.Tensor, obs_full_lengths: int, act_real_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args
            :logits: torch.Tensor of shape (rollout_threads, obs_len + concatenated_action_len, data_dim)
            :obs_full_lengths: int
            :act_real_lengths: torch.Tensor of shape (rollout_threads, num_agents)

        Returns
            :sliced_logits: torch.Tensor of shape (rollout_threads, num_agents, max_new_tokens, data_dim)
        """
        sliced_logits = torch.zeros(act_real_lengths.shape[0], act_real_lengths.shape[1], self.max_new_tokens, logits.shape[-1]).to(logits.device)
        for thread_idx in range(act_real_lengths.shape[0]):
            for agent_idx in range(act_real_lengths.shape[1]):
                if agent_idx == 0:
                    start_idx = obs_full_lengths - 1
                    end_idx = obs_full_lengths + act_real_lengths[thread_idx, agent_idx] - 1
                else:
                    start_idx = end_idx + 1
                    end_idx = start_idx + act_real_lengths[thread_idx, agent_idx]
                sliced_logits[thread_idx, agent_idx, : act_real_lengths[thread_idx, agent_idx]] = logits[thread_idx, start_idx:end_idx].clone()
        return sliced_logits
    
    def get_action_values(self, obs: np.ndarray) -> torch.Tensor:
        """
        Args:
            obs: np.ndarray of shape (rollout_threads, num_agents)

        Returns:
            action_values: torch.Tensor of shape (rollout_threads, num_agents, 1)
        """
        rollout_threads, num_agents = obs.shape
        all_values = []
        for agent_idx in range(self.num_agents):
            token_seq = self.tokenizer(obs[:, agent_idx].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=self.context_window)
            input_ids = token_seq["input_ids"].cuda()
            attn_mask = token_seq["attention_mask"].cuda()

            with self.agent_model.disable_adapter():
                values = self.critic(input_ids, attention_mask=attn_mask).unsqueeze(-1)
            all_values.append(values)
        all_values = torch.cat(all_values, dim=1)
        return all_values
    
    def get_token_values(self, obs: np.ndarray, action_tokens: torch.Tensor, train: bool = False) -> torch.Tensor:
        """
        Args:
            obs: np.ndarray of shape (rollout_threads/batch_size, num_agents)
            action_tokens: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens)

        Returns:
            token_values: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens, data_dim)
        """
        rollout_threads, num_agents = obs.shape
        all_values = []

        for agent_idx in range(num_agents):
            token_seq = self.tokenizer(obs[:, agent_idx].tolist(), return_tensors="pt", padding=True, max_length=self.context_window, truncation=True)
            obs_input_ids = token_seq["input_ids"].cuda()
            obs_attn_mask = token_seq["attention_mask"].cuda()
            obs_full_lengths = obs_input_ids.shape[1]

            act_attn_mask = (action_tokens[:, agent_idx] != self.tokenizer.pad_token_id)
            act_real_lengths = act_attn_mask.sum(dim=-1, keepdim=True)

            obs_act_ids = torch.cat([obs_input_ids, action_tokens[:, agent_idx]], dim=-1)
            obs_act_mask = torch.cat([obs_attn_mask, act_attn_mask], dim=-1)

            with self.agent_model.disable_adapter():
                values = self.critic(obs_act_ids, attention_mask=obs_act_mask)
            token_values = self.get_slice(values, obs_full_lengths, act_real_lengths)
            all_values.append(token_values)
        all_values = torch.cat(all_values, dim=1) # stack on agent dimension
        return all_values

    def get_token_logits(
            self, 
            obs: np.ndarray, 
            action_tokens: torch.Tensor, 
            agent_index: int = None,
            batch_infer: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args
            :obs: np.ndarray of shape (rollout_threads/batch_size, num_agents)
            :action_tokens: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens)

        Returns
            :pi_logits: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens, vocab_size)
            :rho_logits: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens, vocab_size)
        """
        rollout_threads, num_agents = obs.shape
        pi_logits, rho_logits = [], []

        for agent_idx in range(num_agents):
            if agent_index is not None and agent_idx != agent_index:
                continue
            token_seq = self.tokenizer(obs[:, agent_idx].tolist(), return_tensors="pt", padding=True, max_length=self.context_window, truncation=True)
            obs_input_ids = token_seq["input_ids"].cuda()
            obs_attn_mask = token_seq["attention_mask"].cuda()
            obs_full_lengths = obs_input_ids.shape[1]

            act_attn_mask = (action_tokens[:, agent_idx] != self.tokenizer.pad_token_id)
            act_real_lengths = act_attn_mask.sum(dim=-1, keepdim=True)

            obs_act_ids = torch.cat([obs_input_ids, action_tokens[:, agent_idx]], dim=-1)
            obs_act_mask = torch.cat([obs_attn_mask, act_attn_mask], dim=-1)

            if batch_infer:
                pass
            else:
                with self.agent_model.disable_adapter():
                    rho_outputs = self.agents(input_ids=obs_act_ids, attention_mask=obs_act_mask)
                    rho_logits.append(self.get_slice(rho_outputs.logits, obs_full_lengths, act_real_lengths))
                self.agent_model.set_adapter(self.profiles[agent_idx]["role"])
                pi_outputs = self.agents(input_ids=obs_act_ids, attention_mask=obs_act_mask)
                pi_logits.append(self.get_slice(pi_outputs.logits, obs_full_lengths, act_real_lengths))
        rho_logits = torch.cat(rho_logits, dim=1)
        pi_logits = torch.cat(pi_logits, dim=1)
        return pi_logits, rho_logits

    @torch.no_grad()
    def batch_infer(self, model, input_ids, attn_mask, obs_full_lengths, act_real_lengths, infer_batch_size=16,):
        logits = []
        for i in range(0, input_ids.shape[0], infer_batch_size):
            input_ids_batch = input_ids[i : i + infer_batch_size, :]
            attn_mask_batch = attn_mask[i : i + infer_batch_size, :]
            outputs = model(input_ids=input_ids_batch, attention_mask=attn_mask_batch, return_dict=True,)
            logits_batch = self.get_slice(outputs.logits, obs_full_lengths, act_real_lengths)
            logits.append(logits_batch.clone())
        logits = torch.cat(logits, dim=0)
        return logits

    def get_last_token_position(self, action_tokens: torch.Tensor) -> int:
        """
        Given the action tokens, return the last token position.

        Args:
            action_tokens: (torch.Tensor): (max_new_tokens)

        Return:
            last_token_position: (torch.Tensor): int
        """
        pos = len(action_tokens) - 1
        while action_tokens[pos] == self.tokenizer.pad_token_id: pos -= 1
        return pos

    def normalize_log_probs(self, action_log_probs: torch.Tensor, action_token_slice: torch.Tensor) -> torch.Tensor:
        """
        Normalize the log probs by the number of tokens in the action sequence.

        Args:
            action_log_probs: torch.Tensor of shape (1)
            action_token_slice: torch.Tensor of shape (token_length)
        """
        if self.normalization_mode == "token":
            token_length = self.get_last_token_position(action_token_slice) + 1
            action_log_probs /= token_length
        elif self.normalization_mode == "word":
            word_num = len(self.tokenizer.decode(action_token_slice, skip_special_tokens=True).split())
            action_log_probs /= word_num
        elif self.normalization_mode == "sum":
            pass
        else:
            raise NotImplementedError
        return action_log_probs

    def get_joint_action_log_probs(self, obs: np.ndarray, action_tokens: torch.Tensor, agent_to_train: int = None, batch_infer: bool = False):
        """
        Args:
            obs: np.ndarray of shape (rollout_threads/batch_size, num_agents)
            action_tokens: torch.Tensor of shape (rollout_threads/batch_size, num_agents, max_new_tokens)

        Return:
            action_log_probs: torch.Tensor of shape (rollout_threads/batch_size, num_agents)
            entropies: torch.Tensor of shape (rollout_threads/batch_size, num_agents)
        """
        logits, _ = self.get_token_logits(obs, action_tokens, agent_index=agent_to_train, batch_infer=batch_infer)
        # pi_logits: shape (rollout_threads/batch_size, num_agents, max_new_tokens, vocab_size)
        pi_log_softmax = torch.log_softmax(logits, dim=-1)
        log_probs = torch.empty(logits.shape[0], logits.shape[1]).to(logits.device)
        entropies = torch.empty(logits.shape[0], logits.shape[1]).to(logits.device)
        for thread in range(logits.shape[0]):
            for agent in range(self.num_agents):
                if agent_to_train is None:
                    act_token_length = self.get_last_token_position(action_tokens[thread, agent]) + 1
                    log_softmax_slice = pi_log_softmax[thread, agent, :act_token_length, :]
                    action_token_slice = action_tokens[thread, agent, :act_token_length]
                    token_log_probs = torch.gather(log_softmax_slice, -1, action_token_slice.unsqueeze(-1)).squeeze(-1)
                    action_log_prob = self.normalize_log_probs(token_log_probs.sum(), action_token_slice)
                    log_probs[thread, agent] = action_log_prob
                    entropy = Categorical(logits=logits[thread, agent, :act_token_length, :]).entropy().mean()
                    entropies[thread, agent] = entropy
                else:
                    if agent == agent_to_train:
                        act_token_length = self.get_last_token_position(action_tokens[thread, agent]) + 1
                        log_softmax_slice = pi_log_softmax[thread, 0, :act_token_length, :]
                        action_token_slice = action_tokens[thread, agent, :act_token_length]
                        token_log_probs = torch.gather(log_softmax_slice, -1, action_token_slice.unsqueeze(-1)).squeeze(-1)
                        action_log_prob = self.normalize_log_probs(token_log_probs.sum(), action_token_slice)
                        log_probs[thread, 0] = action_log_prob
                        entropy = Categorical(logits=logits[thread, 0, :act_token_length, :]).entropy().mean()
                        entropies[thread, 0] = entropy
                    else:
                        continue
        return log_probs, entropies

    @torch.no_grad()
    def infer_for_rollout(self, obs):
        rollout_obs, rollout_actions, rollout_action_tokens = self.get_actions_sequential(obs)
        if self.algo == "APPO":
            rollout_values = self.get_action_values(rollout_obs)
            rollout_values = rollout_values.float().cpu().numpy()
            action_log_probs, _ = self.get_joint_action_log_probs(rollout_obs, rollout_action_tokens, batch_infer=False)
            rollout_action_tokens = rollout_action_tokens.int().cpu().numpy()
            rollout_log_probs = action_log_probs.float().cpu().numpy()
        elif self.algo == "TPPO":
            rollout_values = self.get_token_values(rollout_obs, rollout_action_tokens).squeeze(-1)
            logits, _ = self.get_token_logits(rollout_obs, rollout_action_tokens)
            logp_softmax = torch.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(logp_softmax, -1, rollout_action_tokens.unsqueeze(-1)).squeeze(-1)
            rollout_values = rollout_values.float().cpu().numpy()
            rollout_action_tokens = rollout_action_tokens.int().cpu().numpy()
            rollout_log_probs = token_log_probs.float().cpu().numpy()
        else:
            raise NotImplementedError

        return rollout_obs, rollout_actions, rollout_action_tokens, rollout_values, rollout_log_probs

    def get_next_tppo_values(self, obs: np.ndarray):
        """
        Args:
            obs: np.ndarray of shape (rollout_threads, num_agents)

        Returns:
            values: torch.Tensor of shape (rollout_threads, num_agents, 1)
        """
        token_seq = self.tokenizer(obs[:, 0].tolist(), return_tensors="pt", padding=True)
        input_ids = token_seq["input_ids"].cuda()
        attn_mask = token_seq["attention_mask"].cuda()

        # values
        with self.agent_model.disable_adapter():
            values = self.critic(input_ids, attention_mask=attn_mask)[:, -1]
        return values

    def get_next_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Get value function predictions.
        Args:
            obs: np.ndarray of shape (rollout_threads, num_agents)

        Returns:
            next_values: np.ndarray of shape (rollout_threads, num_agents, 1)
        """
        if self.algo == "APPO":
            next_action_values = self.get_action_values(obs)
            next_values = next_action_values.cpu().float().numpy()
        elif self.algo == "TPPO":
            next_token_values = self.get_next_tppo_values(obs)
            next_values = next_token_values.cpu().float().numpy()
        else:
            raise NotImplementedError
        return next_values

    def save(self, save_dir: str, steps: int) -> None:
        exp_path = os.path.join(save_dir, "steps_{:04d}".format(steps))
        os.makedirs(exp_path, exist_ok=True)
        self.agents.save_pretrained(exp_path)
        self.critic.save_value_head(os.path.join(exp_path, f"value_head.pth"))
        print(f"[MAS] MAS checkpoints saved → {exp_path}")

    def train(self):
        self.agents.train()
        self.critic.train()

    def eval(self):
        self.agents.eval()
        self.critic.eval()

    @torch.no_grad()
    def collect_actions(self, obs: np.ndarray):
        """Collect trajectories without computing logits or values."""
        rollout_obs, rollout_actions, rollout_action_tokens = self.get_actions_sequential(obs)
        rollout_action_tokens = rollout_action_tokens.int().cpu().numpy()
        return rollout_obs, rollout_actions, rollout_action_tokens

    @torch.no_grad()
    def compute_training_signals(self, obs: np.ndarray, action_tokens: torch.Tensor):
        """Compute values and log probabilities for training."""
        if self.algo == "APPO":
            values = self.get_action_values(obs)
            log_probs, _ = self.get_joint_action_log_probs(obs, action_tokens, batch_infer=False)
            return values.float().cpu().numpy(), log_probs.float().cpu().numpy()
        elif self.algo == "TPPO":
            values = self.get_token_values(obs, action_tokens).squeeze(-1)
            logits, _ = self.get_token_logits(obs, action_tokens)
            logp_softmax = torch.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(logp_softmax, -1, action_tokens.unsqueeze(-1)).squeeze(-1)
            return values.float().cpu().numpy(), token_log_probs.float().cpu().numpy()
        else:
            raise NotImplementedError

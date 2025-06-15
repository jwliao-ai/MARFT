from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig


class Agent:
    """Single MAS agent wrapping a base model and its LoRA adapter."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        profile: dict,
        device: torch.device,
        load_path: Optional[str] = None,
        load_in_4bit: bool = False,
        bf16: bool = True,
        device_map=None,
    ) -> None:
        self.profile = profile
        self.role = profile.get("role", "agent")
        self.device = torch.device(device)

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
            torch_dtype=torch.bfloat16 if bf16 else "auto",
            quantization_config=nf4_config,
            device_map=device_map,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.base_model.enable_input_require_grads()

        if load_path is None:
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.base_model, config)
        else:
            adapter_path = os.path.join(load_path, self.role)
            self.model = PeftModel.from_pretrained(
                self.base_model, adapter_path, adapter_name=self.role
            )
        
        self.model.print_trainable_parameters()
        self.model.half()
        self.model.to(self.device)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device: torch.device):
        self.device = torch.device(device)
        self.model.to(self.device)
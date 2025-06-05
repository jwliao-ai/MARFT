import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class InferenceEngine:
    """Simple inference engine with optional vLLM backend."""
    def __init__(self, model_path: str, max_new_tokens: int, use_vllm: bool = False, tensor_parallel_size: int = 1):
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        if use_vllm:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as e:
                raise ImportError("vLLM is not installed") from e
            self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
            self.sampling_params = SamplingParams(max_tokens=max_new_tokens)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.device = self.model.device

    def generate(self, prompts):
        """Generate sequences for a batch of prompts."""
        if self.use_vllm:
            outputs = self.llm.generate(prompts, self.sampling_params)
            results = []
            for out in outputs:
                results.append(out.outputs[0].text)
            return results
        else:
            tokens = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.generate(**tokens, do_sample=True, top_k=50, temperature=0.5, max_new_tokens=self.max_new_tokens, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
            results = []
            for i in range(len(prompts)):
                generated = outputs[i][tokens['input_ids'][i].shape[0]:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                results.append(text)
            return results

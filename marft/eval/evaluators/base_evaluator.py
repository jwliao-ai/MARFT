import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from datetime import datetime
import numpy as np
import os
from marft.mas import MAS

class BaseEvaluator(ABC):

    def __init__(
            self, 
            mas: MAS, 
            data_path: str | os.PathLike = None, 
            output_dir: str | os.PathLike = None,
            metrics_filename: str = None,
            metrics_timestamp: bool = False,
            response_filename: str = None,
            **kwargs: Dict[str, Any]
        ):
        self.mas = mas
        self.responses = []
        self.results = []
        self.metrics = {}
        self.dataset = self.load_data(data_path)
        self.output_dir = output_dir
        self.metrics_filename = metrics_filename
        self.metrics_timestamp = metrics_timestamp
        self.response_filename = response_filename
        self.args = kwargs

    def load_data(self, data_path: str | os.PathLike):
        with open(data_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries from {data_path}")
        return data

    @abstractmethod
    def evaluate(self):
        """evaluate logics"""
        pass

    def _save_metrics(self):
        if not self.metrics:
            print("⚠️ No metrics to save.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"metrics_{timestamp}.json" if self.metrics_timestamp else "metrics.json"
        output_path = os.path.join(self.output_dir, filename)
        sanitized_metrics = {
            k: float(v) if isinstance(v, (torch.Tensor, np.generic)) else v 
            for k, v in self.metrics.items()
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sanitized_metrics, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"✅ Evaluation metrics saved to {output_path}.")

    def _save_responses(self):
        if not self.responses:
            print("⚠️ No responses to save.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, self.response_filename or "responses.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.responses, f, indent=2, ensure_ascii=False, default=lambda x: str(x))
        print(f"\n✅ Successfully saved {len(self.responses)} responses to {output_file}.")
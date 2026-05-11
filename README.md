<h1 align="center">
MARFT: Multi-Agent Reinforcement Fine-Tuning
</h1>

<p align="center">
| <a href="https://arxiv.org/abs/2504.16129"><b>Paper</b></a> | <a href="https://github.com/SII-MARFT/MARFT"><b>Code</b></a> |
</p>

MARFT is a framework for **multi-agent cooperative reinforcement fine-tuning** of large
language models. It enables teams of LLM agents (e.g., planner, solver, verifier) to
collaborate via shared conversation history and be trained end-to-end with PPO/GRPO
using centralized credit assignment (CTDE — Centralized Training Decentralized
Execution).

Built on top of [AReaL](https://github.com/inclusionAI/AReaL) (v0.5.3), a large-scale
asynchronous RL training system developed by the AReaL Team at Tsinghua IIIS and Ant
Group.

**Key Features**

- **Multi-Agent Workflows**: Sequential DAG and dynamic LLM-orchestrated agent topologies
- **Per-Agent LoRA**: Shared or independent adapter weights per agent role
- **Flexible Critics**: Shared (CTDE) or per-agent LoRA critic modes
- **Credit Assignment**: Equal, step-discounted, or per-step reward distribution strategies

## Getting Started

```bash
git clone https://github.com/SII-MARFT/MARFT.git
cd MARFT
pip install uv
uv sync --extra cuda
```

### Multi-Agent Training (DeepCoder)

```bash
python3 examples/marft/deepcoder_marft.py \
    --config examples/marft/deepcoder_marft_2agent.yaml \
    scheduler.type=local
```

### Multi-Agent Training (DeepScaleR)

```bash
python3 examples/marft/deepscaler_marft.py \
    --config examples/marft/deepscaler_marft_2agent.yaml \
    scheduler.type=local
```

For comprehensive setup and multi-node instructions, see the
[AReaL quickstart guide](https://inclusionai.github.io/AReaL/tutorial/quickstart.html).

## Multi-Agent Training Modes

### Agent Roles

| Agents | Roles (sequential graph)                   |
| ------ | ------------------------------------------ |
| 2      | planner -> solver                          |
| 3      | planner -> solver -> verifier              |
| 4      | planner -> solver -> reflector -> verifier |

### LoRA Modes

| Mode      | Config                                   | Effect                                  |
| --------- | ---------------------------------------- | --------------------------------------- |
| Shared    | `use_multi_lora=true, shared_lora=true`  | All agents share one LoRA adapter       |
| Per-agent | `use_multi_lora=true, shared_lora=false` | Each agent gets its own adapter         |

### Critic Modes

| Mode        | Config                            | Effect                                   |
| ----------- | --------------------------------- | ---------------------------------------- |
| CTDE        | `independent_critic=null`         | Single shared critic (default)           |
| Critic LoRA | `independent_critic=lora`         | Per-agent LoRA adapters on critic        |

## Examples

Example configs and entry scripts are in [`examples/marft/`](examples/marft/):

| Benchmark | Entry Script | Configs |
| --------- | ------------ | ------- |
| **DeepCoder** | `deepcoder_marft.py` | `deepcoder_marft_{2,3,4}agent.yaml`, `deepcoder_marft_{2,3,4}agent_anonymous.yaml` |
| **DeepScaleR** | `deepscaler_marft.py` | `deepscaler_marft_{2,3,4}agent.yaml`, `deepscaler_marft_{2,3,4}agent_anonymous.yaml` |

The `_anonymous` variants run agents without specialized role names or system prompts,
serving as an ablation baseline.

## Acknowledgments

This project is built on [AReaL](https://github.com/inclusionAI/AReaL), developed by
the AReaL Team at Tsinghua IIIS and Ant Group. We gratefully acknowledge their work on
the distributed RL training infrastructure that MARFT extends.

We also appreciate the broader open-source community, including
[ReaLHF](https://github.com/openpsi-project/ReaLHF),
[DeepScaleR](https://github.com/agentica-project/deepscaler),
[SGLang](https://github.com/sgl-project/sglang), and
[vLLM](https://github.com/vllm-project/vllm).

## Citation

If you use MARFT in your research, please cite:

```bibtex
@misc{liao2025marft,
      title={MARFT: Multi-Agent Reinforcement Fine-Tuning},
      author={Junwei Liao and Muning Wen and Jun Wang and Weinan Zhang},
      year={2025},
      eprint={2504.16129},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2504.16129},
}
```

Please also cite the underlying AReaL system:

```bibtex
@misc{fu2025areal,
      title={AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning},
      author={Wei Fu and Jiaxuan Gao and Xujie Shen and Chen Zhu and Zhiyu Mei and Chuyi He and Shusheng Xu and Guo Wei and Jun Mei and Jiashu Wang and Tongkai Yang and Binhang Yuan and Yi Wu},
      year={2025},
      eprint={2505.24298},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24298},
}
```

```bibtex
@inproceedings{mei2025real,
  author       = {Mei, Zhiyu and Fu, Wei and Li, Kaiwei and Wang, Guangju and Zhang, Huanchen and Wu, Yi},
  title        = {ReaL: Efficient RLHF Training of Large Language Models with Parameter Reallocation},
  booktitle    = {Proceedings of the Eighth Conference on Machine Learning and Systems,
                  MLSys 2025, Santa Clara, CA, USA, May 12-15, 2025},
  publisher    = {mlsys.org},
  year         = {2025},
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for
details. MARFT is built upon [AReaL](https://github.com/inclusionAI/AReaL); see
[NOTICE](NOTICE) for original copyright attribution.

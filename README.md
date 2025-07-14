# MARFT: Multi-Agent Reinforcement Fine-Tuning
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jwliao-ai/MARFT/blob/main/LICENSE)

NOTICE: MARFT code has been moved to [SII-MARFT/MARFT](https://github.com/SII-MARFT/MARFT) repository. Following updates will be made there and this repository will keep up to date to that repository.

MARFT stands for **Multi-Agent Reinforcement Fine-Tuning**. This repository implements an LLM-based multi-agent reinforcement fine-tuning framework for general agentic tasks, providing a foundational MARFT framework.

**Check out our paper [MARFT: Multi-Agent Reinforcement Fine-Tuning](https://arxiv.org/abs/2504.16129)!!!**

## Table of Contents
- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
- [Environment Extension](#environment-extension)
- [Multi-Adapter](#multi-adapter)
- [Agent-by-Agent Training](#agent-by-agent-training)
- [Resume Training](#resume-training)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## About
This repository aims to help researchers in academia and industry transition into the world of reinforcement learning. The power of multi-agent systems is vast and often surprising, which is why we provide a comprehensive framework for **MARFT**. The framework supports both **action-level optimization** and **token-level optimization**. It is designed to scale to various agentic tasks by allowing users to craft new environments tailored to their specific needs.

## Features
<!-- - **MAT (Multi-Agent Transformer) Implementation**: A flexible framework for multi-agent reinforcement learning. -->
- **Action and Token Optimization**: Supports both action-level and token-level optimization.
- **Environment Extension**: Easy-to-use tools for creating custom environments for agentic tasks.
- **Multi-Adapter Support**: Agents use the same base model but have different LoRA adapters.
- **Agent-by-Agent Training**: Training individual agents while freezing others for efficient learning.
- **Resume Training**: Resume training from an existing checkpoint.

## Getting Started

### Installation
1. Create a virtual environment:
   ```bash
   conda create -n marft
   conda activate marft
   ```

2. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/jwliao-ai/MARFT.git
   cd MARFT
   pip install -r requirements.txt
   ```

   **Note**: You may need to adjust package versions to match your CUDA version.

<!-- ## Usage
The repository provides a complete framework for MARFT. To get started:
1. Explore the example environment for solving math problems.
2. Use the pre-implemented MAT framework to experiment with different agentic tasks. -->

## Environment Extension
To create a custom environment for your specific agentic task:
1. Navigate to `marft/envs` and create a folder for your environment.
2. Create a Python file (e.g., `env_name.py`) and implement the necessary environment components:
   - `__init__`: Initialize the environment.
   - `reset`: Reset the environment state.
   - `step`: Define the agent's action step.
   - `transition`: Define state transitions.
3. Create a corresponding `runner` and `train` entry in `runner/shared` and `scripts` respectively.

**Example**:
   ```python
   class CustomEnv:
       def __init__(self):
           # Initialize your environment
           pass

       def reset(self):
           # Reset the environment state
           pass

       def step(self, action):
           # Define how the environment responds to actions
           pass

       def transition(self, state):
           # Define state transitions
           pass
   ```

## Multi-Adapter
The framework supports a multi-agent system (MAS) where each agent shares the same base model but uses different **LoRA (Low-Rank Adaptation)** adapters. This allows agents to specialize in different tasks while maintaining a shared foundation. Checkpoint loading is also supported for seamless model resumption.

## Agent-by-Agent Training
The repository supports **agent-by-agent training**, where a single agent is trained while others are frozen. This is controlled by the `--agent_iteration_interval` argument, which defines the training interval for each agent.

## Resume Training
LLMs are hard to train and the training process often crashes if the LLM explores some exotic tokens, which is really normal. Thus, resume training helps to resume training if the LaMAS performance starts to collapse. To use resume training, specify the argument `--load_path`, and under the path, there should be multiple folders contain different LoRA adapter parameters and configurations. Also, a critic model `critic.pth` should be contained and it will be auto-loaded.

## Contributing
We welcome contributions to improve the framework. To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/new-feature`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to your branch: `git push origin feature/new-feature`.
5. Submit a pull request.

## License
This project is licensed under the MIT License. For more details, see the [LICENSE](https://github.com/jwliao-ai/MARFT/blob/main/LICENSE) file.

## Citation
If you find this repository helpful, please consider citing our paper:

```bibtex
@misc{liao2025marftmultiagentreinforcementfinetuning,
      title={MARFT: Multi-Agent Reinforcement Fine-Tuning}, 
      author={Junwei Liao and Muning Wen and Jun Wang and Weinan Zhang},
      year={2025},
      eprint={2504.16129},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2504.16129}, 
}
```

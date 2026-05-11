# Training a Multi-Agent GSM8K Math Team in AReaL

Files in this folder present an example that trains a cooperative multi-agent team on
GSM8K math reasoning, starting from `Qwen/Qwen2.5-1.5B-Instruct`. Three agents
(planner → solver → verifier) share a single policy and pass a shared message history
through a sequential workflow graph. PPO with a centralized critic handles credit
assignment (CTDE pattern).

## Architecture

| Agent | Role |
| --- | --- |
| **Planner** | Breaks the problem into numbered steps without solving it. |
| **Solver** | Follows the plan and produces a `\boxed{}` answer. |
| **Verifier** | Checks the answer; restates it if correct, corrects it otherwise. |

All three share a backbone model (and optional LoRA adapter). The team reward is
computed by comparing the final `\boxed{}` answer against the ground truth via
`MathVerifyWorker`. Credit assignment distributes the reward across agent boundaries
using a configurable strategy (`equal`, `step_discount`, or `per_step`).

## To run the example

```bash
python3 examples/multi_agent_math/gsm8k_ma_rl.py \
    --config examples/multi_agent_math/gsm8k_ma_ppo.yaml \
    scheduler.type=ray \
    experiment_name=gsm8k-multi-agent trial_name=trial0
```

## Key config differences from single-agent GRPO

The `multi_agent:` section in `gsm8k_ma_ppo.yaml` introduces:

```yaml
multi_agent:
  graph_type: sequential
  role_names: [planner, solver, verifier]
  role_configs:
    planner: { system_prompt: "...", description: "..." }
    solver:  { system_prompt: "...", description: "..." }
    verifier: { system_prompt: "...", description: "..." }
  credit_strategy: equal      # or step_discount / per_step
  credit_discount: 1.0
  shared_policy: true
  dump_transcripts: true
```

The training backbone is PPO (actor + critic + reference) rather than GRPO, so the
config includes `critic:` and `ref:` sections.

## Distributed compatibility

The entry script uses the **string path + kwargs** pattern for workflow instantiation,
matching the existing `RLVRWorkflow` and `MultiturnRLVRWorkflow` examples. Each
distributed worker reconstructs the workflow from serializable arguments:

- `reward_fn` is passed as an importable string path
- `tokenizer` is passed as a HuggingFace model path
- `graph` and `roles` are plain dicts

Eval uses a separate gconfig with lower temperature (`0.6`) and `n_samples=1`.

## Customization

- **Add/remove agents**: Edit `role_names` and `role_configs`. The graph auto-chains
  them sequentially.
- **Parallel fan-out**: Provide a `graph_config` with explicit nodes/edges instead of
  `graph_type: sequential`.
- **Credit strategy**: Set `credit_strategy: step_discount` with `credit_discount: 0.9`
  to give earlier agents a discounted share of the team reward.
- **Per-role generation**: Override `max_new_tokens`, `temperature`, or `top_p` per role
  inside `role_configs`.
- **Transition messages**: Add `transition_messages: [null, "Now solve it.", "Now verify."]`
  to inject custom instructions between agents.

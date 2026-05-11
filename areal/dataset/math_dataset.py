"""MATH dataset loaders (hendrycks/math).

The MATH dataset is stored as JSONL files with fields:
``problem``, ``solution``, ``answer``, ``subject``, ``level``, ``unique_id``.

For RL training, the ``problem`` field is wrapped into the chat-message
format expected by rollout workflows, while ``answer`` is preserved for
the reward function.
"""

from datasets import load_dataset


def get_math_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """Load the MATH dataset for RL training.

    Reads JSONL files from *path* (expects ``train.jsonl`` / ``test.jsonl``),
    maps each sample into the standard ``messages`` + ``answer`` schema
    required by rollout workflows and reward functions.
    """
    data_files = {"train": "train.jsonl", "test": "test.jsonl"}
    dataset = load_dataset("json", data_dir=path, data_files=data_files, split=split)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["problem"]
                + "\nPlease put your final answer within \\boxed{}.",
            }
        ]
        return {"messages": messages, "answer": sample["answer"]}

    columns_to_remove = [
        c for c in dataset.column_names if c not in ("messages", "answer")
    ]
    dataset = dataset.map(process).remove_columns(columns_to_remove)

    if max_length is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


def get_math_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """Load the MATH dataset for SFT training.

    Concatenates ``problem`` + ``solution`` into a single sequence with
    loss masking on the prompt portion.
    """
    data_files = {"train": "train.jsonl", "test": "test.jsonl"}
    dataset = load_dataset("json", data_dir=path, data_files=data_files, split=split)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["problem"] + sample["solution"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["problem"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    columns_to_remove = [
        c for c in dataset.column_names if c not in ("input_ids", "loss_mask")
    ]
    dataset = dataset.map(process).remove_columns(columns_to_remove)

    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset

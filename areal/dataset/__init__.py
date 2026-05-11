from typing import TYPE_CHECKING, Optional

from areal.api.cli_args import _DatasetConfig
from areal.utils import logging

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

VALID_DATASETS = [
    "gsm8k",
    "MATH",
    "clevr_count_70k",
    "geometry3k",
    "hh-rlhf",
    "torl_data",
    "deepcoder",
    "deepscaler",
    "aime_2024",
    "MATH-500",
    "olympiadbench",
    "minervamath",
    "codeforces",
]

logger = logging.getLogger("Dataset")


def _get_custom_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    if "gsm8k" in path and type == "sft":
        from .gsm8k import get_gsm8k_sft_dataset

        return get_gsm8k_sft_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "gsm8k" in path and type == "rl":
        from .gsm8k import get_gsm8k_rl_dataset

        return get_gsm8k_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "MATH-500" in path and type == "rl":
        from .deepscaler import get_math500_rl_dataset

        return get_math500_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "MATH" in path and type == "rl":
        from .math_dataset import get_math_rl_dataset

        return get_math_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "MATH" in path and type == "sft":
        from .math_dataset import get_math_sft_dataset

        return get_math_sft_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "clevr_count_70k" in path and type == "sft":
        from .clevr_count_70k import get_clevr_count_70k_sft_dataset

        return get_clevr_count_70k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "clevr_count_70k" in path and type == "rl":
        from .clevr_count_70k import get_clevr_count_70k_rl_dataset

        return get_clevr_count_70k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "geometry3k" in path and type == "sft":
        from .geometry3k import get_geometry3k_sft_dataset

        return get_geometry3k_sft_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "geometry3k" in path and type == "rl":
        from .geometry3k import get_geometry3k_rl_dataset

        return get_geometry3k_rl_dataset(
            path=path,
            split=split,
            processor=processor,
            max_length=max_length,
            **kwargs,
        )
    elif "hh-rlhf" in path and type == "rw":
        from .hhrlhf import get_hhrlhf_rw_dataset

        return get_hhrlhf_rw_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "torl_data" in path and type == "rl":
        from .torl_data import get_torl_data_rl_dataset

        return get_torl_data_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "deepcoder" in path.lower() and type == "rl":
        from .deepcoder import get_deepcoder_rl_dataset

        return get_deepcoder_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif ("DeepScaleR" in path or "deepscaler" in path.lower()) and type == "rl":
        from .deepscaler import get_deepscaler_rl_dataset

        return get_deepscaler_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "aime_2024" in path and type == "rl":
        from .deepscaler import get_aime2024_rl_dataset

        return get_aime2024_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "olympiadbench" in path and type == "rl":
        from .deepscaler import get_olympiadbench_rl_dataset

        return get_olympiadbench_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "minervamath" in path and type == "rl":
        from .deepscaler import get_minervamath_rl_dataset

        return get_minervamath_rl_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    elif "codeforces" in path.lower() and type == "rl" and split in ("train", None):
        from .codeforces import get_codeforces_train_dataset

        return get_codeforces_train_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
        )
    elif "codeforces" in path.lower() and type == "rl":
        from .codeforces import get_codeforces_test_dataset

        return get_codeforces_test_dataset(
            path=path,
            split=split,
            tokenizer=tokenizer,
            max_length=max_length,
        )
    else:
        raise ValueError(
            f"Dataset {path} with split {split} and training type {type} is not supported. "
            f"Supported datasets are: {VALID_DATASETS}. "
        )


def get_custom_dataset(
    split: str | None = None,
    dataset_config: _DatasetConfig | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    if dataset_config is not None:
        return _get_custom_dataset(
            path=dataset_config.path,
            type=dataset_config.type,
            split=split,
            max_length=dataset_config.max_length,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs,
        )

    logger.warning("dataset_config is not provided")
    return _get_custom_dataset(
        split=split,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )

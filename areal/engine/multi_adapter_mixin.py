"""Multi-adapter LoRA mixin for FSDP training engines.

Provides ``MultiAdapterMixin``, a mixin class that extends a single-adapter
PEFT/LoRA engine (e.g. ``FSDPPPOActor``) with the ability to manage multiple
named LoRA adapters.  This is useful for multi-agent training scenarios where
each agent owns a dedicated adapter and is trained on only that agent's tokens.

Usage::

    class MultiAdapterPPOActor(MultiAdapterMixin, FSDPPPOActor):
        ...
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, TaskType
from torch import nn

from areal.api.io_struct import WeightUpdateMeta
from areal.utils import logging

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine

logger = logging.getLogger("MultiAdapterMixin")


class MultiAdapterMixin:
    """Mixin that adds multi-adapter LoRA support to FSDP engines.

    Intended to be composed via MRO *before* an ``FSDPEngine`` subclass so
    that ``self.model``, ``self.config``, and weight-sync helpers are
    available.  The mixin never calls ``super().__init__`` — all
    initialisation happens explicitly through :meth:`setup_multi_adapter`.
    """

    _adapter_names: list[str]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_multi_adapter(self, adapter_names: list[str]) -> None:
        """Register additional named LoRA adapters on the PEFT model.

        This must be called **after** the base engine's ``initialize()``
        (which invokes ``_apply_peft_wrapper`` and creates the ``"default"``
        adapter).  Each name in *adapter_names* that is not ``"default"``
        will be added via ``model.add_adapter``.

        Args:
            adapter_names: Ordered list of adapter names.  The index in this
                list is used as the adapter index when interpreting
                ``agent_ids`` tensors.
        """
        if not adapter_names:
            raise ValueError("adapter_names must be a non-empty list")

        config = self.config  # type: ignore[attr-defined]

        if not config.target_modules or config.target_modules == ["all-linear"]:
            target_modules = "all-linear"
        else:
            target_modules = config.target_modules

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            bias="none",
        )

        model = self.model  # type: ignore[attr-defined]
        for name in adapter_names:
            if name == "default":
                continue
            model.add_adapter(name, lora_config)
            logger.info(f"Added LoRA adapter '{name}' (r={config.lora_rank})")

        self._adapter_names = list(adapter_names)
        logger.info(
            f"Multi-adapter setup complete: {self._adapter_names} "
            f"({len(self._adapter_names)} adapters)"
        )

    def setup_multi_adapter_critic(self, adapter_names: list[str]) -> None:
        """Register per-agent LoRA adapters for a critic (TokenClassification) model.

        Like :meth:`setup_multi_adapter` but uses ``TaskType.TOKEN_CLS`` and
        ``modules_to_save=["score"]`` so the value head stays fully trainable
        while transformer layers get per-agent LoRA adapters.
        """
        if not adapter_names:
            raise ValueError("adapter_names must be a non-empty list")

        config = self.config  # type: ignore[attr-defined]

        if not config.target_modules or config.target_modules == ["all-linear"]:
            target_modules = "all-linear"
        else:
            target_modules = config.target_modules

        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["score"],
        )

        model = self.model  # type: ignore[attr-defined]
        for name in adapter_names:
            if name == "default":
                continue
            model.add_adapter(name, lora_config)
            logger.info(
                f"Added critic LoRA adapter '{name}' "
                f"(r={config.lora_rank}, modules_to_save=['score'])"
            )

        self._adapter_names = list(adapter_names)
        logger.info(
            f"Multi-adapter critic setup complete: {self._adapter_names} "
            f"({len(self._adapter_names)} adapters)"
        )

    # ------------------------------------------------------------------
    # Adapter queries
    # ------------------------------------------------------------------

    def get_adapter_names(self) -> list[str]:
        """Return the ordered list of adapter names."""
        return list(self._adapter_names)

    def set_active_adapter(self, adapter_name: str) -> None:
        """Activate a specific adapter for the next forward pass."""
        self.model.set_adapter(adapter_name)  # type: ignore[attr-defined]

    def iter_adapter_params(
        self, adapter_name: str
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Yield ``(name, param)`` pairs for a specific adapter's trainable parameters.

        Only parameters whose name contains the adapter identifier **and**
        that require gradients are yielded.  For the ``"default"`` adapter
        the filter string is ``"default"`` (PEFT names default adapter
        weights with this substring).
        """
        model = self.model  # type: ignore[attr-defined]
        for name, param in model.named_parameters():
            if param.requires_grad and adapter_name in name:
                yield name, param

    # ------------------------------------------------------------------
    # Loss-mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_adapter_loss_mask(
        loss_mask: torch.Tensor,
        agent_ids: torch.Tensor,
        adapter_index: int,
    ) -> torch.Tensor:
        """Compute a per-adapter loss mask.

        Args:
            loss_mask: Original loss mask of shape ``[batch, seq_len]``.
            agent_ids: Tensor of shape ``[batch, seq_len]`` mapping each
                token to an adapter/agent index (``-1`` for prompt tokens).
            adapter_index: The adapter index to select.

        Returns:
            A boolean tensor of the same shape where only tokens belonging
            to *adapter_index* (and originally unmasked) are ``True``.
        """
        return loss_mask.bool() & (agent_ids == adapter_index)

    # ------------------------------------------------------------------
    # Weight synchronisation
    # ------------------------------------------------------------------

    def update_multi_adapter_weights(
        self,
        meta: WeightUpdateMeta,
    ) -> None:
        """Broadcast all adapter weights to the connected inference engine.

        Iterates over every registered adapter in order, collects its
        trainable parameters, and pushes them through the existing
        ``_update_bucket_weights_from_distributed`` chunked-broadcast
        mechanism.

        This method mirrors ``_update_weights_from_distributed`` but scopes
        each broadcast to a single adapter's parameters.
        """
        import torch.distributed as dist

        from areal.infra.platforms import current_platform

        meta.nccl_master_address = self.weight_update_master_addr  # type: ignore[attr-defined]
        meta.nccl_master_port = self.weight_update_master_port  # type: ignore[attr-defined]
        meta.nccl_group_name = self.weight_update_group_name  # type: ignore[attr-defined]

        rollout_engine: InferenceEngine = self.rollout_engine  # type: ignore[attr-defined]
        cpu_group = self.cpu_group  # type: ignore[attr-defined]

        if dist.get_rank() == 0:
            rollout_engine.pause_generation()

        dist.barrier(group=cpu_group)

        main_rank = dist.get_rank() == 0
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

        for adapter_name in self._adapter_names:
            self.set_active_adapter(adapter_name)

            buffer_size = 0
            named_tensors: list[tuple[str, torch.Tensor]] = []

            for name, param in self._get_model_name_parameters():  # type: ignore[attr-defined]
                if not (param.requires_grad and adapter_name in name):
                    continue
                tensor = self._get_full_tensor(param)  # type: ignore[attr-defined]

                if not main_rank:
                    continue

                tensor_size = tensor.numel() * tensor.element_size()
                if tensor_size + buffer_size > weight_chunked_mem_size:
                    self._update_bucket_weights_from_distributed(  # type: ignore[attr-defined]
                        meta, named_tensors
                    )
                    buffer_size = 0

                named_tensors.append((name, tensor))
                buffer_size += tensor_size

            if named_tensors:
                self._update_bucket_weights_from_distributed(  # type: ignore[attr-defined]
                    meta, named_tensors
                )

            logger.info(f"Synced adapter '{adapter_name}' weights to inference engine")

        dist.barrier(group=cpu_group)

        if dist.get_rank() == 0:
            rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=cpu_group)

    def update_adapter_weights(
        self,
        meta: WeightUpdateMeta,
        adapter_name: str,
    ) -> None:
        """Broadcast a single adapter's weights to the inference engine.

        Args:
            meta: Weight update metadata (must be ``"xccl"`` type).
            adapter_name: Name of the adapter whose parameters to sync.
        """
        import torch.distributed as dist

        from areal.infra.platforms import current_platform

        if adapter_name not in self._adapter_names:
            raise ValueError(
                f"Unknown adapter '{adapter_name}'. "
                f"Registered adapters: {self._adapter_names}"
            )

        meta.nccl_master_address = self.weight_update_master_addr  # type: ignore[attr-defined]
        meta.nccl_master_port = self.weight_update_master_port  # type: ignore[attr-defined]
        meta.nccl_group_name = self.weight_update_group_name  # type: ignore[attr-defined]

        rollout_engine: InferenceEngine = self.rollout_engine  # type: ignore[attr-defined]
        cpu_group = self.cpu_group  # type: ignore[attr-defined]

        if dist.get_rank() == 0:
            rollout_engine.pause_generation()

        dist.barrier(group=cpu_group)

        self.set_active_adapter(adapter_name)

        main_rank = dist.get_rank() == 0
        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in self._get_model_name_parameters():  # type: ignore[attr-defined]
            if not (param.requires_grad and adapter_name in name):
                continue
            tensor = self._get_full_tensor(param)  # type: ignore[attr-defined]

            if not main_rank:
                continue

            tensor_size = tensor.numel() * tensor.element_size()
            if tensor_size + buffer_size > weight_chunked_mem_size:
                self._update_bucket_weights_from_distributed(  # type: ignore[attr-defined]
                    meta, named_tensors
                )
                buffer_size = 0

            named_tensors.append((name, tensor))
            buffer_size += tensor_size

        if named_tensors:
            self._update_bucket_weights_from_distributed(  # type: ignore[attr-defined]
                meta, named_tensors
            )

        dist.barrier(group=cpu_group)

        if dist.get_rank() == 0:
            rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=cpu_group)

        logger.info(f"Synced adapter '{adapter_name}' weights to inference engine")

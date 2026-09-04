"""Qwen2MoE adapter with tiered memory placement and expert paging."""

from __future__ import annotations

from typing import Any
from dataclasses import replace

import torch

from sparse_llm.models.adapters import TransformersCausalLMAdapter, ModelCapabilities


def is_qwen2_moe_config(config: Any) -> bool:
    """Detect Qwen2MoE model from config."""
    return getattr(config, "model_type", None) == "qwen2_moe"


class Qwen2MoEAdapter(TransformersCausalLMAdapter):
    """Qwen2MoE adapter with tiered memory and expert paging support."""

    def _discover_capabilities(self, config: Any | None, model: Any | None) -> ModelCapabilities:
        """Override to enable expert_paging for Qwen2MoE models."""
        capabilities = super()._discover_capabilities(config, model)
        if config is not None and getattr(config, "model_type", None) == "qwen2_moe":
            # Enable expert paging for Qwen2MoE
            capabilities = replace(capabilities, expert_paging=True)
        return capabilities


__all__ = ["Qwen2MoEAdapter", "is_qwen2_moe_config"]

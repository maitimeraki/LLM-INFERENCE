"""Safe compatibility adapter for models with unknown MoE topology."""

from __future__ import annotations

from sparse_llm.models.adapters import TransformersCausalLMAdapter


class UniversalMoEAdapter(TransformersCausalLMAdapter):
    """Use ordinary Transformers execution until paging is proven."""

    pass


__all__ = ["UniversalMoEAdapter"]

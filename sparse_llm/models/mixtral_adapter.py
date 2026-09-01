"""Mixtral-8x7B adapter with MoE paging metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from sparse_llm.models.adapters import TransformersCausalLMAdapter, ModelCapabilities
from sparse_llm.models.paging import PagingCapabilities
from sparse_llm.models.shared_weight_loader import SharedWeightPlacer


def is_mixtral_config(config: Any) -> bool:
    """Detect Mixtral model from config."""
    return getattr(config, "model_type", None) == "mixtral"


@dataclass(frozen=True)
class MixtralPagingMetadata:
    """Router and expert tensor names for Mixtral architecture."""

    num_hidden_layers: int
    num_local_experts: int
    num_experts_per_tok: int

    @property
    def router_tensor_names(self) -> tuple[str, ...]:
        """Generate router tensor names for each layer."""
        return tuple(
            f"model.layers.{i}.block_sparse_moe.gate" for i in range(self.num_hidden_layers)
        )

    @property
    def expert_tensor_names(self) -> tuple[str, ...]:
        """Generate expert tensor names for all experts in all layers."""
        names = []
        for layer in range(self.num_hidden_layers):
            for expert in range(self.num_local_experts):
                for weight_name in ("w1", "w2", "w3"):
                    names.append(
                        f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{weight_name}"
                    )
        return tuple(names)


class MixtralAdapter(TransformersCausalLMAdapter):
    """Mixtral-8x7B adapter with paging-aware topology extraction."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self._paging_metadata: MixtralPagingMetadata | None = None
        super().__init__(model_id, **kwargs)
        if self.config is not None:
            self._extract_paging_metadata()

    def _extract_paging_metadata(self) -> None:
        """Extract Mixtral-specific paging metadata from config."""
        if self.config is None:
            return
        num_layers = getattr(self.config, "num_hidden_layers", None)
        num_experts = getattr(self.config, "num_local_experts", None)
        num_experts_per_tok = getattr(self.config, "num_experts_per_tok", None)
        if num_layers is not None and num_experts is not None and num_experts_per_tok is not None:
            self._paging_metadata = MixtralPagingMetadata(
                num_hidden_layers=num_layers,
                num_local_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
            )

    def _discover_capabilities(self, config: Any | None, model: Any | None) -> ModelCapabilities:
        """Override to enable expert_paging for Mixtral models."""
        capabilities = super()._discover_capabilities(config, model)
        # Check if this is a Mixtral model and extract metadata
        if config is not None and getattr(config, "model_type", None) == "mixtral":
            num_layers = getattr(config, "num_hidden_layers", None)
            num_experts = getattr(config, "num_local_experts", None)
            num_experts_per_tok = getattr(config, "num_experts_per_tok", None)
            if num_layers is not None and num_experts is not None and num_experts_per_tok is not None:
                capabilities = replace(capabilities, expert_paging=True)
        return capabilities

    def load(self) -> "MixtralAdapter":
        """Load model and extract paging metadata, classify weights."""
        super().load()
        self._extract_paging_metadata()
        # Classify weights for memory-safe loading (diagnostic only, no device placement)
        if self.config is not None:
            placer = SharedWeightPlacer(self.config)
            # Classification is available via placer.classify_tensor(name)
            # Future integration: use for device placement decisions
        return self

    @property
    def paging_capabilities(self) -> PagingCapabilities | None:
        """Return Mixtral paging capabilities with tensor metadata."""
        if self._paging_metadata is None:
            return None
        return PagingCapabilities.from_mixtral(self._paging_metadata)


__all__ = ["MixtralAdapter", "MixtralPagingMetadata", "is_mixtral_config"]

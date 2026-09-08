"""Model architecture introspection and weight size estimation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    """Information about model architecture and weight distribution."""
    model_id: str
    is_moe: bool
    num_layers: int
    num_experts: int
    num_experts_per_tok: int
    shared_weight_bytes: int
    expert_weight_bytes: int  # Per expert
    total_bytes: int
    # Added for accurate KV cache calculation
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int


class ModelIntrospector:
    """Analyze model config to detect architecture and estimate weight sizes."""

    def introspect(self, model_id: str) -> ModelInfo:
        """Load config from Hugging Face and introspect.

        Args:
            model_id: Hugging Face model ID or local path

        Returns:
            ModelInfo with architecture details
        """
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        return self.introspect_from_config(config, model_id=model_id)

    def introspect_from_config(self, config: Any, model_id: str) -> ModelInfo:
        """Introspect from loaded config object.

        Args:
            config: Transformers config object
            model_id: Model identifier for logging

        Returns:
            ModelInfo with architecture details
        """
        # Detect MoE architecture
        is_moe, num_experts, num_experts_per_tok = self._detect_moe(config)

        # Get layer count
        num_layers = getattr(config, "num_hidden_layers", 0)

        # Estimate weight sizes
        shared_weight_bytes = self._estimate_shared_weights(config, is_moe, num_layers)
        expert_weight_bytes = self._estimate_expert_weights(config) if is_moe else 0

        # Total size
        total_bytes = shared_weight_bytes
        if is_moe:
            total_bytes += expert_weight_bytes * num_experts * num_layers
        else:
            # Dense model: FFN weights are "shared" (always loaded)
            ffn_bytes = self._estimate_dense_ffn_weights(config)
            total_bytes += ffn_bytes * num_layers

        # Extract architecture parameters for KV cache calculation
        hidden_size = getattr(config, "hidden_size", 4096)
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
        num_attention_heads = getattr(config, "num_attention_heads", 32)
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

        return ModelInfo(
            model_id=model_id,
            is_moe=is_moe,
            num_layers=num_layers,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            shared_weight_bytes=shared_weight_bytes,
            expert_weight_bytes=expert_weight_bytes,
            total_bytes=total_bytes,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads
        )

    def _detect_moe(self, config: Any) -> tuple[bool, int, int]:
        """Detect if model is MoE and extract expert count.

        Returns:
            (is_moe, num_experts, num_experts_per_tok)
        """
        # Try multiple attribute names (different architectures use different names)
        num_experts = 0
        for attr in ["num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"]:
            try:
                val = getattr(config, attr, None)
                if val is not None and isinstance(val, int) and val > 0:
                    num_experts = val
                    break
            except AttributeError:
                continue

        num_experts_per_tok = 0
        for attr in ["num_experts_per_tok", "num_experts_per_token", "top_k", "moe_top_k"]:
            try:
                val = getattr(config, attr, None)
                if val is not None and isinstance(val, int) and val > 0:
                    num_experts_per_tok = val
                    break
            except AttributeError:
                continue

        is_moe = num_experts > 1 and num_experts_per_tok > 0

        return is_moe, num_experts, num_experts_per_tok

    def _estimate_shared_weights(self, config: Any, is_moe: bool, num_layers: int) -> int:
        """Estimate total bytes for shared weights (embeddings, attention, norms, router)."""
        hidden_size = getattr(config, "hidden_size", 4096)
        vocab_size = getattr(config, "vocab_size", 32000)

        # Embeddings: vocab_size * hidden_size * 2 bytes (fp16)
        embeddings_bytes = vocab_size * hidden_size * 2

        # Per-layer attention: 4 projections * hidden_size^2 * 2 bytes
        attention_bytes_per_layer = 4 * hidden_size * hidden_size * 2

        # Per-layer norms: 2 norms * hidden_size * 2 bytes (small)
        norm_bytes_per_layer = 2 * hidden_size * 2

        # Router (MoE only): hidden_size * num_experts * 2 bytes per layer
        router_bytes_per_layer = 0
        if is_moe:
            num_experts = getattr(config, "num_local_experts", 8)
            router_bytes_per_layer = hidden_size * num_experts * 2

        total_shared = embeddings_bytes + num_layers * (
            attention_bytes_per_layer + norm_bytes_per_layer + router_bytes_per_layer
        )

        return total_shared

    def _estimate_expert_weights(self, config: Any) -> int:
        """Estimate bytes for one expert's weights."""
        hidden_size = getattr(config, "hidden_size", 4096)
        intermediate_size = getattr(config, "intermediate_size", 14336)

        # Expert FFN: 3 projections (w1, w2, w3 or gate_proj, up_proj, down_proj)
        # w1/gate_proj: hidden_size * intermediate_size
        # w2/down_proj: intermediate_size * hidden_size
        # w3/up_proj: hidden_size * intermediate_size

        expert_bytes = (
            hidden_size * intermediate_size * 2  # w1
            + intermediate_size * hidden_size * 2  # w2
            + hidden_size * intermediate_size * 2  # w3
        )

        return expert_bytes

    def _estimate_dense_ffn_weights(self, config: Any) -> int:
        """Estimate bytes for dense model FFN per layer."""
        # Same calculation as expert, but for dense FFN
        return self._estimate_expert_weights(config)

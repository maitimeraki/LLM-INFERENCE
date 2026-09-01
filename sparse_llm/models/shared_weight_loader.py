"""Weight classification for memory-safe model loading with shared-weight separation."""

from __future__ import annotations

from typing import Any


class SharedWeightPlacer:
    """Classifies model weights as shared, expert, or other for memory-safe paging."""

    def __init__(self, config: Any) -> None:
        """Initialize with model config for weight pattern generation.

        Args:
            config: Model config object with num_hidden_layers and num_local_experts.
        """
        self.config = config
        self.num_hidden_layers = getattr(config, "num_hidden_layers", 0)
        self.num_local_experts = getattr(config, "num_local_experts", 0)

    def shared_weight_names(self) -> set[str]:
        """Generate set of shared (non-expert) weight tensor names.

        Includes: embeddings, attention, normalization, routers, LM head.
        Returns:
            Set of tensor name patterns for shared weights.
        """
        shared = set()

        # Embeddings
        shared.add("model.embed_tokens.weight")

        # Layer-specific shared weights: attention, layer norms, routers
        for i in range(self.num_hidden_layers):
            # Self-attention weights
            shared.add(f"model.layers.{i}.self_attn.q_proj.weight")
            shared.add(f"model.layers.{i}.self_attn.k_proj.weight")
            shared.add(f"model.layers.{i}.self_attn.v_proj.weight")
            shared.add(f"model.layers.{i}.self_attn.o_proj.weight")

            # Layer norms
            shared.add(f"model.layers.{i}.input_layernorm.weight")
            shared.add(f"model.layers.{i}.post_attention_layernorm.weight")

            # Router (gating network for MoE)
            shared.add(f"model.layers.{i}.block_sparse_moe.gate")

        # Final normalization
        shared.add("model.norm.weight")

        # LM head
        shared.add("lm_head.weight")

        return shared

    def expert_weight_names(self) -> set[str]:
        """Generate set of expert (pageable) weight tensor names.

        Includes: all block_sparse_moe.experts.* tensors.
        Returns:
            Set of tensor name patterns for expert weights.
        """
        experts = set()

        for layer in range(self.num_hidden_layers):
            for expert in range(self.num_local_experts):
                for weight_name in ("w1", "w2", "w3"):
                    experts.add(
                        f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{weight_name}.weight"
                    )

        return experts

    def classify_tensor(self, name: str) -> str:
        """Classify a tensor by name as shared, expert, or other.

        Args:
            name: Tensor name from model state dict.

        Returns:
            Classification: 'shared', 'expert', or 'other'.
        """
        if name in self.shared_weight_names():
            return "shared"
        if name in self.expert_weight_names():
            return "expert"
        return "other"


__all__ = ["SharedWeightPlacer"]

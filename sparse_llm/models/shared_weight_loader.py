"""Weight classification for memory-safe model loading with shared-weight separation."""

from __future__ import annotations

from typing import Any


class SharedWeightPlacer:
    """Classifies model weights as shared, expert, or other for memory-safe paging."""

    def __init__(self, config: Any, *, tied_weights: dict[str, str] | None = None) -> None:
        """Initialize with model config for weight pattern generation.

        Args:
            config: Model config object with num_hidden_layers and num_local_experts.
            tied_weights: Optional mapping of alias names to canonical tensor names.
                         e.g., {"lm_head.weight": "model.embed_tokens.weight"}
        """
        self.config = config
        self.num_hidden_layers = getattr(config, "num_hidden_layers", 0)
        self.num_local_experts = getattr(config, "num_local_experts", 0)
        self._tied_weights = tied_weights or {}

    def shared_weight_names(self) -> set[str]:
        """Generate set of shared (non-expert) weight tensor names.

        Universal approach: Pattern-based detection works for all MoE architectures.
        Includes: embeddings, attention, normalization, routers, LM head, shared experts.
        Returns:
            Set of tensor name patterns for shared weights.
        """
        shared = set()

        # Embeddings (universal across all transformer architectures)
        shared.add("model.embed_tokens.weight")
        shared.add("transformer.wte.weight")  # GPT-style
        shared.add("embeddings.word_embeddings.weight")  # BERT-style

        # Layer-specific shared weights: attention, layer norms, routers
        for i in range(self.num_hidden_layers):
            # Self-attention weights (universal pattern)
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                shared.add(f"model.layers.{i}.self_attn.{proj}.weight")
                shared.add(f"model.layers.{i}.attention.{proj}.weight")  # Alternative naming
                shared.add(f"transformer.h.{i}.attn.{proj}.weight")  # GPT-style

            # Layer norms (universal pattern)
            shared.add(f"model.layers.{i}.input_layernorm.weight")
            shared.add(f"model.layers.{i}.post_attention_layernorm.weight")
            shared.add(f"transformer.h.{i}.ln_1.weight")  # GPT-style
            shared.add(f"transformer.h.{i}.ln_2.weight")

            # MoE Router/Gate (universal patterns - all MoE models have some form)
            # These patterns cover: Mixtral, Qwen2MoE, DeepSeek, Switch Transformers, etc.
            shared.add(f"model.layers.{i}.block_sparse_moe.gate")  # Mixtral pattern
            shared.add(f"model.layers.{i}.mlp.gate.weight")  # Qwen2MoE pattern
            shared.add(f"model.layers.{i}.moe.gate.weight")  # Generic MoE pattern
            shared.add(f"model.layers.{i}.ffn.gate.weight")  # Alternative FFN naming

            # Shared/Always-Active Expert (models like Qwen2MoE, DeepSeek)
            # These are NOT routed, so they go on fastest device with other shared weights
            for weight in ["gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]:
                shared.add(f"model.layers.{i}.mlp.shared_expert.{weight}.weight")
                shared.add(f"model.layers.{i}.moe.shared_expert.{weight}.weight")

        # Final normalization (universal patterns)
        shared.add("model.norm.weight")
        shared.add("transformer.ln_f.weight")  # GPT-style
        shared.add("encoder.final_layernorm.weight")

        # LM head (universal patterns)
        shared.add("lm_head.weight")
        shared.add("transformer.wte.weight")  # Some models tie this
        shared.add("cls.predictions.decoder.weight")

        return shared

    def expert_weight_names(self) -> set[str]:
        """Generate set of expert (pageable) weight tensor names.

        Universal: resolves expert count and weight names directly from config attributes,
        covering all known MoE naming conventions without model_type branching.
        Returns:
            Set of tensor name patterns for expert weights.
        """
        experts = set()

        # Resolve expert count from whichever attribute the model uses
        num_experts = (
            getattr(self.config, "num_local_experts", None)
            or getattr(self.config, "num_experts", None)
            or getattr(self.config, "n_routed_experts", None)
            or getattr(self.config, "num_expert_ffn", None)
            or 0
        )
        if num_experts == 0:
            return experts

        # All known weight projection names across MoE architectures
        # Mixtral: w1/w2/w3  |  Qwen2MoE/LLaMA-MoE/DeepSeek: gate_proj/up_proj/down_proj
        # Switch Transformer: wi/wo  |  Phi-MoE: fc1/fc2
        projection_names = ("w1", "w2", "w3", "gate_proj", "up_proj", "down_proj", "wi", "wo", "fc1", "fc2")

        # All known namespace paths where experts live
        # Mixtral: block_sparse_moe.experts  |  Qwen2MoE/DeepSeek: mlp.experts
        # Generic/custom: moe.experts, ffn.experts
        expert_namespaces = (
            "block_sparse_moe.experts",
            "mlp.experts",
            "moe.experts",
            "ffn.experts",
        )

        for layer in range(self.num_hidden_layers):
            for namespace in expert_namespaces:
                for expert in range(num_experts):
                    for proj in projection_names:
                        experts.add(f"model.layers.{layer}.{namespace}.{expert}.{proj}.weight")
                        # Some models omit .weight suffix on expert tensors
                        experts.add(f"model.layers.{layer}.{namespace}.{expert}.{proj}")

        return experts

    def canonical_name(self, name: str) -> str:
        """Resolve tied weight aliases to canonical tensor name.

        Args:
            name: Tensor name from model state dict.

        Returns:
            Canonical name (may be the same as input if not an alias).
        """
        return self._tied_weights.get(name, name)

    def is_tied_alias(self, name: str) -> bool:
        """Check if tensor name is a tied weight alias (not canonical).

        Args:
            name: Tensor name from model state dict.

        Returns:
            True if name is an alias that should share storage with canonical tensor.
        """
        return name in self._tied_weights

    def classify_tensor(self, name: str) -> str:
        """Classify a tensor by name as shared, expert, or other.

        Args:
            name: Tensor name from model state dict.

        Returns:
            Classification: 'shared', 'expert', or 'other'.
        """
        canonical = self.canonical_name(name)
        if canonical in self.shared_weight_names():
            return "shared"
        if canonical in self.expert_weight_names():
            return "expert"
        return "other"


__all__ = ["SharedWeightPlacer"]

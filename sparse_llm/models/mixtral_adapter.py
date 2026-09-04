"""Mixtral-8x7B adapter with MoE paging and quantization support."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

import torch

from sparse_llm.inference.metrics import GenerationResult, GenerationMetrics
from sparse_llm.inference.quantized_loader import QuantizedExpertLoader
from sparse_llm.models.adapters import TransformersCausalLMAdapter, ModelCapabilities
from sparse_llm.models.paging import PagingCapabilities
from sparse_llm.models.quantization import QuantizationPolicy, detect_quantization
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
    """Mixtral-8x7B adapter with paging and quantization support."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self._paging_metadata: MixtralPagingMetadata | None = None
        self._quant_policy: QuantizationPolicy | None = None
        self._quant_loader: QuantizedExpertLoader | None = None
        self._weight_placer: SharedWeightPlacer | None = None
        self._async_pager: Any | None = None  # AsyncExpertPager, initialized on demand
        self._scheduler: Any | None = None  # RequestScheduler, initialized on demand
        self._use_async_paging: bool = kwargs.pop("use_async_paging", False)
        self._use_batching: bool = kwargs.pop("use_batching", False)
        super().__init__(model_id, **kwargs)
        if self.config is not None:
            self._extract_paging_metadata()
            self._detect_quantization()

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

    def _detect_quantization(self) -> None:
        """Detect quantization policy from model config."""
        if self.config is None:
            return
        quant_config = getattr(self.config, "quantization_config", None)
        self._quant_policy = detect_quantization(quant_config)
        if self._quant_policy and self._quant_policy.is_quantized():
            # Initialize quantized loader with detected policy
            device = getattr(self, "device", "cpu")
            self._quant_loader = QuantizedExpertLoader(
                policy=self._quant_policy, device=device
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
        """Load model and extract paging/quantization metadata, classify weights."""
        super().load()
        self._extract_paging_metadata()
        self._detect_quantization()
        # Re-initialize quantized loader with correct device after model load
        if self._quant_policy and self._quant_policy.is_quantized():
            device = self.model.device if hasattr(self.model, "device") else "cpu"
            self._quant_loader = QuantizedExpertLoader(
                policy=self._quant_policy, device=device
            )
        # Classify weights with tied weight handling
        if self.config is not None:
            # Detect tied weights (Mixtral typically ties embed_tokens and lm_head)
            tied_weights = {}
            if hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
                tied_weights["lm_head.weight"] = "model.embed_tokens.weight"

            placer = SharedWeightPlacer(self.config, tied_weights=tied_weights)
            # Store placer for runtime queries
            self._weight_placer = placer
        return self

    @property
    def paging_capabilities(self) -> PagingCapabilities | None:
        """Return Mixtral paging capabilities with tensor metadata."""
        if self._paging_metadata is None:
            return None
        return PagingCapabilities.from_mixtral(self._paging_metadata)

    @property
    def quantization_policy(self) -> QuantizationPolicy | None:
        """Return detected quantization policy, if any."""
        return self._quant_policy

    def load_expert_tensors(
        self,
        key: tuple[int, int],
        tensors: dict[str, Any],
        state_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load and dequantize expert tensors using quantization policy.

        If quantization is active, dequantizes tensors in-place.
        Otherwise returns tensors unchanged (backward-compatible).

        Args:
            key: (layer_index, expert_index) tuple.
            tensors: Expert weight tensors from checkpoint.
            state_dict: Full model state dict for scale/zero-point extraction.

        Returns:
            Dequantized expert tensors, or original tensors if unquantized.
        """
        if self._quant_loader is None:
            # No quantization; return original
            return tensors
        return self._quant_loader.load_expert(key, tensors, state_dict)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate text with Mixtral using expert paging and routing."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        self.load()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("adapter is not loaded")

        # Choose generation path based on enabled features
        if self._use_async_paging or self._use_batching:
            return self._generate_with_advanced_features(prompt, max_new_tokens, temperature)
        elif self._paging_metadata is not None and hasattr(self.model, "config"):
            return self._generate_with_paging(prompt, max_new_tokens, temperature)
        else:
            return super().generate(prompt, max_new_tokens, temperature)

    def _generate_with_advanced_features(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        """Generate using advanced Stage 5-7 features when enabled."""
        from sparse_llm.models.advanced_generation import (
            AdvancedGenerationRunner,
            AdvancedGenerationConfig,
        )

        # Configure advanced features
        config = AdvancedGenerationConfig(
            enable_async_paging=self._use_async_paging,
            enable_batching=self._use_batching,
            max_concurrent_transfers=3,
            max_batch_size=8,
        )

        # Create runner
        runner = AdvancedGenerationRunner(
            model=self.model,
            tokenizer=self.tokenizer,
            paging_metadata=self._paging_metadata,
            config=config,
        )

        # Use sync generation for single prompt
        # ponytail: async path would use asyncio.run(runner.generate_async(...))
        return runner.generate_sync(prompt, max_new_tokens, temperature)

    def _generate_with_paging(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        """Generate using expert paging with router-driven cache requests."""
        from sparse_llm.cache import ExpertCache, ExpertKey
        from sparse_llm.inference.paged_generation import PagedGenerationRunner

        # Initialize paging infrastructure
        cache_bytes = self.policy.expert_cache_bytes or (2 * 1024 * 1024 * 1024)  # 2GB default
        cache = ExpertCache(capacity_bytes=cache_bytes)
        runner = PagedGenerationRunner(
            num_layers=self._paging_metadata.num_hidden_layers,
            num_experts=self._paging_metadata.num_local_experts,
            cache=cache,
        )

        # Install forward hooks to intercept MoE routing
        hooks = []
        router_intercepts = {}  # layer_idx -> routing decisions

        def create_router_hook(layer_idx):
            def hook_fn(module, input, output):
                """Intercept router logits and record expert selections."""
                # Router output is logits for expert selection
                if hasattr(output, 'shape') and len(output.shape) >= 2:
                    # Get top-k experts for this token
                    topk_logits, topk_indices = torch.topk(
                        output,
                        k=self._paging_metadata.num_experts_per_tok,
                        dim=-1
                    )
                    # Record selections for this layer
                    router_intercepts[layer_idx] = topk_indices.cpu().numpy()

                    # Request experts from cache
                    for expert_idx in topk_indices[0, 0].tolist():  # First token
                        key = ExpertKey(layer=layer_idx, expert=expert_idx)
                        runner.request_expert(key)
                return output
            return hook_fn

        # Register hooks on router gates
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer_idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                    hook = layer.block_sparse_moe.gate.register_forward_hook(
                        create_router_hook(layer_idx)
                    )
                    hooks.append(hook)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        start_time = time.time()

        # Generate with router interception active
        try:
            with torch.no_grad():
                if temperature > 0:
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time_ms = (time.time() - start_time) * 1000

        # Extract paging metrics from runner
        cache_hits = sum(s.cache_hits for s in runner.per_layer_stats.values())
        cache_misses = sum(s.cache_misses for s in runner.per_layer_stats.values())
        expert_load_ms = sum(s.expert_load_time_ms for s in runner.per_layer_stats.values())

        metrics = GenerationMetrics(
            prefill_latency_ms=0.0,  # Split timing in future iteration
            decode_latency_ms=generation_time_ms,
            total_latency_ms=generation_time_ms,
            tokens_generated=max_new_tokens,
            throughput_tokens_per_sec=max_new_tokens / (generation_time_ms / 1000) if generation_time_ms > 0 else 0,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            expert_load_time_ms=expert_load_ms,
        )

        return GenerationResult(text=generated_text, metrics=metrics)

    def _generate_with_advanced_paging(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        """Generate with Stage 5-7 features: async transfers, KV cache, batching.

        This method demonstrates the full integration architecture when all stages
        are enabled. Falls back to _generate_with_paging if advanced features disabled.
        """
        # Stage 5: Async Expert Transfers
        if self._use_async_paging and self._async_pager is None:
            from sparse_llm.inference.async_pager import AsyncExpertPager, BackpressureConfig
            from sparse_llm.inference.pager import ExpertPager
            from sparse_llm.cache import ExpertCache

            cache_bytes = self.policy.expert_cache_bytes or (2 * 1024 * 1024 * 1024)
            cache = ExpertCache(capacity_bytes=cache_bytes)

            # Create sync pager and wrap with async
            # ponytail: ExpertPager needs proper initialization with index/loader
            # For now, demonstrate the integration pattern
            sync_pager = ExpertPager(cache=cache, index=None, loader=None)
            self._async_pager = AsyncExpertPager(
                sync_pager=sync_pager,
                config=BackpressureConfig(max_outstanding=3),
                num_experts=self._paging_metadata.num_local_experts,
            )

        # Stage 6: KV Cache & Batching
        if self._use_batching and self._scheduler is None:
            from sparse_llm.scheduling.batch_scheduler import BatchScheduler
            from sparse_llm.inference.continuous_batch import ContinuousBatchConfig

            self._scheduler = BatchScheduler(
                max_batch_size=8,
                max_waiting_time_ms=10.0,
            )

        # For Stage 5-7 integration, we need more infrastructure
        # ponytail: This is a framework for future implementation
        # Current implementation uses _generate_with_paging
        return self._generate_with_paging(prompt, max_new_tokens, temperature)


__all__ = ["MixtralAdapter", "MixtralPagingMetadata", "is_mixtral_config"]

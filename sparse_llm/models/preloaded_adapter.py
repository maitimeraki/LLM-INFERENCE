"""Adapter that uses pre-loaded weights from resource-aware loading system."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from sparse_llm.models.adapters import ModelAdapter, ModelCapabilities, GenerationResult, GenerationMetrics
from sparse_llm.loading.loaded_weight_state import LoadedWeightState


class PreloadedWeightAdapter(ModelAdapter):
    """Adapter that wraps pre-loaded weights from LoadedWeightState.

    This adapter bridges resource-aware loaded weights into the inference pipeline
    by constructing a model that uses the pre-loaded weights instead of loading
    from scratch.
    """

    def __init__(self, loaded_state: LoadedWeightState):
        """Initialize adapter with pre-loaded weights.

        Args:
            loaded_state: Pre-loaded weight state from resource-aware loading
        """
        self._loaded_state = loaded_state
        self._model = None
        self._tokenizer = None
        self._config = None
        self._is_loaded = False

    def load(self) -> None:
        """Load model using pre-loaded weights."""
        if self._is_loaded:
            return

        model_id = self._loaded_state.model_info.model_id

        # Load config and tokenizer (lightweight)
        self._config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

        # Create model structure without loading weights
        # We'll inject our pre-loaded weights
        self._model = AutoModelForCausalLM.from_config(self._config)

        # Inject pre-loaded shared weights
        self._inject_shared_weights()

        # Move model to correct device
        device = self._loaded_state.placement_plan.shared_device
        self._model = self._model.to(device)
        self._model.eval()

        self._is_loaded = True

    def _inject_shared_weights(self) -> None:
        """Inject pre-loaded shared weights into model structure."""
        model_state = self._model.state_dict()

        # Replace shared weights with pre-loaded versions
        for name, tensor in self._loaded_state.shared_weights.items():
            if name in model_state:
                # Direct replacement
                model_state[name] = tensor
            else:
                # Try without "model." prefix
                alt_name = name.replace("model.", "", 1)
                if alt_name in model_state:
                    model_state[alt_name] = tensor

        # Load the modified state dict
        self._model.load_state_dict(model_state, strict=False)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        is_moe = self._loaded_state.model_info.is_moe
        return ModelCapabilities(
            expert_paging=is_moe,
            expert_cache_size_bytes=self._loaded_state.expert_cache.gpu_slots *
                                   self._loaded_state.model_info.expert_weight_bytes if is_moe else 0,
            universal_adapter=True
        )

    @property
    def model(self) -> Any:
        """Return the underlying model."""
        return self._model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Generate text using the model with pre-loaded weights.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            GenerationResult with text, tokens, and metrics
        """
        if not self._is_loaded:
            self.load()

        import time
        start_time = time.time()

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = self._loaded_state.placement_plan.shared_device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Track initial cache state
        initial_cache_stats = self._loaded_state.expert_cache.get_stats()

        # Generate
        with torch.no_grad():
            if temperature == 0.0:
                # Greedy decoding
                outputs = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
                )
            else:
                # Sampling
                outputs = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
                )

        # Decode output
        generated_ids = outputs[0][input_ids.shape[1]:]  # Only new tokens
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate metrics
        elapsed_time = time.time() - start_time
        num_tokens = len(generated_ids)
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0.0

        # Get cache statistics
        final_cache_stats = self._loaded_state.expert_cache.get_stats()
        cache_hits = final_cache_stats["gpu_hits"] - initial_cache_stats["gpu_hits"]
        cache_misses = final_cache_stats["storage_hits"] - initial_cache_stats["storage_hits"]

        metrics = GenerationMetrics(
            prefill_time_ms=0.0,  # Not separately tracked
            decode_time_ms=elapsed_time * 1000,
            tokens_per_sec=tokens_per_sec,
            cache_hits=cache_hits if self._loaded_state.model_info.is_moe else 0,
            cache_misses=cache_misses if self._loaded_state.model_info.is_moe else 0,
            expert_load_time_ms=0.0  # Not separately tracked
        )

        return GenerationResult(
            text=full_text,
            token_ids=outputs[0].tolist(),
            metrics=metrics
        )

    def clear_cache(self) -> None:
        """Clear KV cache if present."""
        if self._model is not None and hasattr(self._model, 'clear_cache'):
            self._model.clear_cache()


__all__ = ["PreloadedWeightAdapter"]

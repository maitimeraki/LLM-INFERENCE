"""Single public inference runtime backed by a model adapter."""

from __future__ import annotations

from typing import Any

from sparse_llm.models.adapters import DevicePolicy, ModelAdapter, ModelCapabilities
from sparse_llm.models.registry import ModelRegistry, create_model_adapter, get_default_registry


class InferenceEngine:
    """Run real generation through one model-independent adapter.

    ``model`` may be a Hugging Face identifier/local directory or an already
    constructed adapter, which keeps tests independent of model downloads.
    """

    def __init__(
        self,
        model: str | ModelAdapter | None = None,
        *,
        model_name: str | None = None,
        device: str = "auto",
        dtype: str | object | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        device_map: Any | None = None,
        offload_folder: str | None = None,
        expert_cache_bytes: int | None = None,
        registry: ModelRegistry | None = None,
        adapter: ModelAdapter | None = None,
    ) -> None:
        if model is not None and model_name is not None:
            raise ValueError("pass either model or model_name, not both")
        if adapter is not None and (model is not None or model_name is not None):
            raise ValueError("pass adapter or model, not both")
        if expert_cache_bytes is not None and (
            not isinstance(expert_cache_bytes, int)
            or isinstance(expert_cache_bytes, bool)
            or expert_cache_bytes < 1
        ):
            raise ValueError("expert_cache_bytes must be a positive integer")
        source = model if model is not None else model_name
        if adapter is None:
            if isinstance(source, ModelAdapter):
                adapter = source
            elif isinstance(source, str):
                policy = DevicePolicy(
                    device=device,
                    dtype=dtype,
                    revision=revision,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    expert_cache_bytes=expert_cache_bytes,
                )
                adapter = (registry or get_default_registry()).create(source, policy=policy)
            else:
                raise ValueError("a model identifier/path or adapter is required")
        self.adapter = adapter
        self._last_result = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return discovered capabilities, or provisional defaults before load."""
        return self.adapter.capabilities

    @property
    def model(self) -> Any | None:
        """Expose the loaded model for diagnostics without loading it."""
        return getattr(self.adapter, "model", None)

    def load(self) -> "InferenceEngine":
        """Load model resources and return this engine."""
        self.adapter.load()
        return self

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ):
        """Run real tokenization, prefill, and autoregressive decoding."""
        self._last_result = self.adapter.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return self._last_result

    def stats(self) -> dict[str, Any]:
        """Return measured state without inventing cache or paging metrics."""
        result = self._last_result
        return {
            "capabilities": self.capabilities.to_dict(),
            "loaded": self.model is not None,
            "metrics": result.metrics.to_dict() if result is not None else None,
        }

    def clear_cache(self) -> None:
        """Clear adapter-owned generation cache when an adapter provides it."""
        clear = getattr(self.adapter, "clear_cache", None)
        if clear is not None:
            clear()


__all__ = ["InferenceEngine", "create_model_adapter"]

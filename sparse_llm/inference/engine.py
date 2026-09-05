"""Single public inference runtime backed by a model adapter."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from sparse_llm.models.adapters import DevicePolicy, ModelAdapter, ModelCapabilities
from sparse_llm.models.registry import ModelRegistry, create_model_adapter, get_default_registry

if TYPE_CHECKING:
    from sparse_llm.loading.loaded_weight_state import LoadedWeightState


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
        loaded_state: "LoadedWeightState | None" = None,
    ) -> None:
        """Initialize inference engine.

        Args:
            model: Hugging Face model ID, local path, or ModelAdapter instance
            model_name: Alternative way to specify model (deprecated, use model)
            device: Device placement ("auto", "cuda", "cpu")
            dtype: Model dtype ("float32", "float16", "bfloat16")
            revision: Model revision/commit hash
            local_files_only: Don't download, use cached files only
            trust_remote_code: Allow custom model code execution
            device_map: Transformers device map for multi-GPU
            offload_folder: Folder for CPU offloading
            expert_cache_bytes: Expert cache size in bytes
            registry: Custom model registry
            adapter: Pre-constructed adapter
            loaded_state: Pre-loaded weight state from resource-aware loading.
                         If provided, bypasses traditional model loading.
        """
        if model is not None and model_name is not None:
            raise ValueError("pass either model or model_name, not both")
        if adapter is not None and (model is not None or model_name is not None):
            raise ValueError("pass adapter or model, not both")
        if loaded_state is not None and (model is not None or model_name is not None or adapter is not None):
            raise ValueError("loaded_state cannot be combined with model or adapter")
        if expert_cache_bytes is not None and (
            not isinstance(expert_cache_bytes, int)
            or isinstance(expert_cache_bytes, bool)
            or expert_cache_bytes < 1
        ):
            raise ValueError("expert_cache_bytes must be a positive integer")

        # Handle resource-aware loaded state
        if loaded_state is not None:
            self._initialize_from_loaded_state(loaded_state)
            return

        # Traditional loading path
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
                # Pre-load config so registry can select correct specialization (e.g., Qwen2MoEAdapter)
                try:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(
                        source,
                        revision=revision,
                        local_files_only=local_files_only,
                        trust_remote_code=trust_remote_code,
                    )
                except Exception:
                    config = None
                adapter = (registry or get_default_registry()).create(source, policy=policy, config=config)
            else:
                raise ValueError("a model identifier/path or adapter is required")
        self.adapter = adapter
        self._last_result = None
        self._loaded_state = None

    def _initialize_from_loaded_state(self, state: "LoadedWeightState") -> None:
        """Initialize from resource-aware loaded state.

        Args:
            state: Pre-loaded weight state
        """
        from sparse_llm.models.preloaded_adapter import PreloadedWeightAdapter

        # Create adapter that uses pre-loaded weights
        self.adapter = PreloadedWeightAdapter(state)
        self._loaded_state = state
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

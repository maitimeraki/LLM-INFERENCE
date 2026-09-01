"""Model-independent adapters for local causal language-model inference.

Transformers is intentionally imported only inside the loading method so that
package imports, tests, and CLI help remain usable without model dependencies.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import torch

from sparse_llm.inference.metrics import GenerationMetrics, GenerationResult
from sparse_llm.models.paging import PagingCapabilities, PagingValidationResult


ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class DevicePolicy:
    """Explicit model loading and execution options."""

    device: str = "auto"
    dtype: str | torch.dtype | None = None
    revision: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    device_map: str | Mapping[str, int] | None = None
    offload_folder: str | None = None
    expert_cache_bytes: int | None = None

    def resolve_device(self) -> torch.device:
        """Resolve and validate the requested execution device."""
        if not isinstance(self.device, str):
            raise ValueError(f"unsupported device {self.device!r}")
        requested = self.device.lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        try:
            return torch.device(requested)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"unsupported device {self.device!r}") from error

    def torch_dtype(self) -> torch.dtype | None:
        """Convert a friendly dtype name into a PyTorch dtype."""
        if self.dtype is None or isinstance(self.dtype, torch.dtype):
            return self.dtype
        names = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        try:
            return names[self.dtype.lower()]
        except (AttributeError, KeyError) as error:
            raise ValueError(f"unsupported dtype {self.dtype!r}") from error


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities discovered from a loaded model, never guessed by name."""

    model_id: str
    model_type: str | None = None
    architecture: str | None = None
    architecture_classification: str = "unknown"
    is_moe: bool = False
    supports_generation: bool = True
    supports_kv_cache: bool = True
    expert_paging: bool = False
    num_hidden_layers: int | None = None
    num_experts: int | None = None
    top_k_experts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable capability data for diagnostics."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "architecture_classification": self.architecture_classification,
            "is_moe": self.is_moe,
            "supports_generation": self.supports_generation,
            "supports_kv_cache": self.supports_kv_cache,
            "expert_paging": self.expert_paging,
            "num_hidden_layers": self.num_hidden_layers,
            "num_experts": self.num_experts,
            "top_k_experts": self.top_k_experts,
        }


class ModelAdapter(ABC):
    """Contract shared by generic and validated architecture adapters."""

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return capabilities discovered for this model."""

    @property
    def paging_capabilities(self) -> PagingCapabilities | None:
        """Return validated paging evidence, if this adapter provides it."""
        return None

    @property
    def paging_adapter(self) -> "ModelAdapter | None":
        """Return this adapter if it provides paging capabilities, else None."""
        if self.paging_capabilities is not None:
            return self
        return None

    def validate_paging(self) -> PagingValidationResult:
        """Explain why this adapter can or cannot activate expert paging."""
        capabilities = self.paging_capabilities
        if capabilities is None:
            return PagingValidationResult.unavailable(
                "adapter has no architecture-validated expert paging contract"
            )
        return PagingValidationResult.from_capabilities(capabilities)

    @abstractmethod
    def load(self) -> "ModelAdapter":
        """Load tokenizer/model resources lazily and return this adapter."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Run real autoregressive generation for one prompt."""


class TransformersCausalLMAdapter(ModelAdapter):
    """Generic adapter for any Transformers-compatible causal LM.

    The adapter uses the model's forward implementation and KV cache for a
    simple, deterministic greedy/sampling decode. It deliberately does not
    claim expert paging: generic model internals are not safe to mutate.
    """

    def __init__(
        self,
        model_id: str,
        policy: DevicePolicy | None = None,
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
        config: Any | None = None,
        tokenizer_loader: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty model identifier or path")
        self.model_id = model_id
        self.policy = policy or DevicePolicy()
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self._tokenizer_loader = tokenizer_loader
        self._model_loader = model_loader
        self._capabilities = ModelCapabilities(model_id=model_id)
        if config is not None:
            self._capabilities = self._discover_capabilities(config, model)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def load(self) -> "TransformersCausalLMAdapter":
        """Load tokenizer and model only when the caller starts inference."""
        if self.tokenizer is None or self.model is None:
            tokenizer_loader, model_loader = self._loaders()
            common = {
                "revision": self.policy.revision,
                "local_files_only": self.policy.local_files_only,
                "trust_remote_code": self.policy.trust_remote_code,
            }
            common = {key: value for key, value in common.items() if value is not None}
            if self.tokenizer is None:
                self.tokenizer = tokenizer_loader.from_pretrained(self.model_id, **common)
            if self.model is None:
                model_kwargs = dict(common)
                dtype = self.policy.torch_dtype()
                if dtype is not None:
                    model_kwargs["torch_dtype"] = dtype
                if self.policy.device_map is not None:
                    model_kwargs["device_map"] = self.policy.device_map
                if self.policy.offload_folder is not None:
                    model_kwargs["offload_folder"] = self.policy.offload_folder
                self.model = model_loader.from_pretrained(self.model_id, **model_kwargs)

        if self.model is None or self.tokenizer is None:  # pragma: no cover - loader contract
            raise RuntimeError("model and tokenizer loaders returned no objects")
        if self.policy.device_map is None:
            target = self.policy.resolve_device()
            if hasattr(self.model, "to"):
                self.model.to(target)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.config = self.config or getattr(self.model, "config", None)
        self._capabilities = self._discover_capabilities(self.config, self.model)
        return self

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Tokenize and decode a prompt using real model forward passes."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.load()
        if self.model is None or self.tokenizer is None:  # pragma: no cover
            raise RuntimeError("adapter is not loaded")

        encoded = self._encode(prompt)
        input_ids, attention_mask = self._input_tensors(encoded)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("tokenizer must return one non-empty sequence")
        input_ids = input_ids.to(self._input_device())
        attention_mask = attention_mask.to(input_ids.device)
        full_ids = input_ids.clone()
        prompt_tokens = int(input_ids.shape[1])
        eos_ids = self._eos_ids()
        peak_started = self._start_memory_measurement()

        with torch.inference_mode():
            prefill_start = time.perf_counter()
            output = self._forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            self._synchronize()
            prefill_ms = (time.perf_counter() - prefill_start) * 1000
            past = self._output_value(output, "past_key_values")
            logits = self._output_value(output, "logits")[:, -1, :]

            decode_start = time.perf_counter()
            generated = 0
            for _ in range(max_new_tokens):
                next_token = self._select_token(logits, temperature)
                full_ids = torch.cat((full_ids, next_token[:, None]), dim=1)
                generated += 1
                if eos_ids and int(next_token.item()) in eos_ids:
                    break
                attention_mask = torch.cat(
                    (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)),
                    dim=1,
                )
                if past is None:
                    step_ids = full_ids
                else:
                    step_ids = next_token[:, None]
                output = self._forward(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = self._output_value(output, "past_key_values")
                logits = self._output_value(output, "logits")[:, -1, :]
            self._synchronize()
            decode_ms = (time.perf_counter() - decode_start) * 1000
        peak_vram = self._finish_memory_measurement(peak_started)

        token_ids = [int(token) for token in full_ids[0].tolist()]
        text = self._decode(token_ids)
        decode_seconds = decode_ms / 1000
        return GenerationResult(
            text=text,
            token_ids=token_ids,
            metrics=GenerationMetrics(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated,
                prefill_latency_ms=prefill_ms,
                decode_latency_ms=decode_ms,
                throughput_tokens_per_sec=(generated / decode_seconds if decode_seconds else 0.0),
                peak_vram_bytes=peak_vram,
                # Generic generation never swaps experts, so these remain honest zeros.
                cache_hits=0,
                cache_misses=0,
                expert_load_time_ms=0.0,
            ),
        )

    def _loaders(self) -> tuple[Any, Any]:
        if self._tokenizer_loader is not None and self._model_loader is not None:
            return self._tokenizer_loader, self._model_loader
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Transformers is required to load a model; install the project runtime dependencies"
            ) from error
        return AutoTokenizer, AutoModelForCausalLM

    def _discover_capabilities(self, config: Any | None, model: Any | None) -> ModelCapabilities:
        if config is None:
            return replace(self._capabilities, architecture_classification="unknown")
        model_type = getattr(config, "model_type", None)
        architectures = getattr(config, "architectures", None) or []
        architecture = architectures[0] if architectures else type(model).__name__ if model is not None else None
        expert_count = self._first_int(config, "num_local_experts", "num_experts", "n_routed_experts")
        top_k = self._first_int(config, "num_experts_per_tok", "num_experts_per_token", "num_selected_experts", "top_k")
        layers = self._first_int(config, "num_hidden_layers", "n_layer", "num_layers")
        is_moe = expert_count is not None or top_k is not None
        classification = "moe" if is_moe else "dense"
        supports_kv = bool(getattr(config, "use_cache", True))
        return ModelCapabilities(
            model_id=self.model_id,
            model_type=model_type,
            architecture=architecture,
            architecture_classification=classification,
            is_moe=is_moe,
            supports_generation=True,
            supports_kv_cache=supports_kv,
            expert_paging=False,
            num_hidden_layers=layers,
            num_experts=expert_count,
            top_k_experts=top_k,
        )

    @staticmethod
    def _first_int(source: Any, *names: str) -> int | None:
        for name in names:
            value = getattr(source, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    def _encode(self, prompt: str) -> Any:
        try:
            return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        except TypeError:
            return self.tokenizer(prompt, return_tensors="pt")

    @staticmethod
    def _input_tensors(encoded: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(encoded, Mapping):
            input_ids = encoded.get("input_ids")
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = getattr(encoded, "input_ids", None)
            attention_mask = getattr(encoded, "attention_mask", None)
        if input_ids is None:
            raise ValueError("tokenizer output does not contain input_ids")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        return input_ids, attention_mask

    def _input_device(self) -> torch.device:
        if self.policy.device_map is not None:
            try:
                return next(self.model.parameters()).device
            except (AttributeError, StopIteration):
                pass
        return self.policy.resolve_device()

    def _forward(self, **kwargs: Any) -> Any:
        try:
            return self.model(**kwargs, return_dict=True)
        except TypeError as error:
            if "return_dict" not in str(error):
                raise
            kwargs.pop("return_dict", None)
            return self.model(**kwargs)

    @staticmethod
    def _output_value(output: Any, name: str, default: Any = None) -> Any:
        if isinstance(output, Mapping):
            return output.get(name, default)
        value = getattr(output, name, None)
        if value is not None:
            return value
        if isinstance(output, (tuple, list)):
            if name == "logits" and output:
                return output[0]
            if name == "past_key_values" and len(output) > 1:
                return output[1]
            return default
        try:
            return output[name]
        except (KeyError, IndexError, TypeError):
            if default is not None:
                return default
            raise ValueError(f"model output does not contain {name}") from None

    @staticmethod
    def _select_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 0:
            return torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    def _eos_ids(self) -> set[int]:
        value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            value = getattr(self.config, "eos_token_id", None)
        if isinstance(value, int):
            return {value}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return {int(item) for item in value}
        return set()

    def _decode(self, token_ids: list[int]) -> str:
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            return self.tokenizer.decode(token_ids)

    def _start_memory_measurement(self) -> bool:
        device = self._input_device()
        if device.type != "cuda":
            return False
        torch.cuda.reset_peak_memory_stats(device)
        self._synchronize()
        return True

    def _finish_memory_measurement(self, started: bool) -> int:
        if not started:
            return 0
        self._synchronize()
        return int(torch.cuda.max_memory_allocated(self._input_device()))

    def _synchronize(self) -> None:
        device = self._input_device()
        if device.type == "cuda":
            torch.cuda.synchronize(device)


__all__ = [
    "DevicePolicy",
    "ExpertKey",
    "ModelAdapter",
    "ModelCapabilities",
    "TransformersCausalLMAdapter",
]

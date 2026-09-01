"""Registry for generic and validated model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sparse_llm.models.adapters import (
    DevicePolicy,
    ModelAdapter,
    TransformersCausalLMAdapter,
)
from sparse_llm.models.mixtral_adapter import MixtralAdapter, is_mixtral_config


AdapterFactory = Callable[..., ModelAdapter]
AdapterPredicate = Callable[[Any], bool]


@dataclass(frozen=True)
class _Registration:
    name: str
    factory: AdapterFactory
    predicate: AdapterPredicate
    priority: int


class ModelRegistry:
    """Select validated specializations before the generic fallback.

    A specialization predicate receives checkpoint configuration only. A
    missing configuration always selects the generic adapter.
    """

    def __init__(self, *, include_generic: bool = True) -> None:
        self._registrations: list[_Registration] = []
        self._generic_enabled = include_generic

    def register(
        self,
        name: str,
        factory: AdapterFactory,
        predicate: AdapterPredicate,
        *,
        priority: int = 0,
    ) -> None:
        """Register a configuration-validated specialization."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("adapter name must be non-empty")
        if not callable(factory) or not callable(predicate):
            raise TypeError("factory and predicate must be callable")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise ValueError("adapter priority must be an integer")
        if any(item.name == name for item in self._registrations):
            raise ValueError(f"adapter {name!r} is already registered")
        self._registrations.append(_Registration(name, factory, predicate, priority))
        self._registrations.sort(key=lambda item: item.priority, reverse=True)

    def create(
        self,
        model_id: str,
        policy: DevicePolicy | None = None,
        *,
        config: Any | None = None,
        **kwargs: Any,
    ) -> ModelAdapter:
        """Create an adapter without loading model artifacts."""
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty model identifier or path")
        if config is not None:
            for registration in self._registrations:
                if registration.predicate(config):
                    return registration.factory(model_id, policy=policy, config=config, **kwargs)
        if not self._generic_enabled:
            raise ValueError(f"no adapter registered for model {model_id!r}")
        return TransformersCausalLMAdapter(model_id, policy=policy, config=config, **kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered specialization names in selection order."""
        return tuple(item.name for item in self._registrations)


_DEFAULT_REGISTRY = ModelRegistry()
_DEFAULT_REGISTRY.register(
    "mixtral",
    lambda model_id, *, policy=None, config=None, **kwargs: MixtralAdapter(
        model_id, policy=policy, config=config, **kwargs
    ),
    is_mixtral_config,
    priority=10,
)


def get_default_registry() -> ModelRegistry:
    """Return the process-local registry used by the public runtime."""
    return _DEFAULT_REGISTRY


def create_model_adapter(
    model_id: str,
    policy: DevicePolicy | None = None,
    **kwargs: Any,
) -> ModelAdapter:
    """Create the default adapter without loading model artifacts."""
    return _DEFAULT_REGISTRY.create(model_id, policy=policy, **kwargs)


__all__ = ["ModelRegistry", "create_model_adapter", "get_default_registry"]

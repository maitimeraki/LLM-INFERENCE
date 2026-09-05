"""Public API for model-independent local inference and cache primitives."""

from sparse_llm.cache import ExpertCache, ExpertKey
from sparse_llm.inference.metrics import GenerationMetrics, GenerationResult
from sparse_llm.integrations import UnifiedMemoryCoordinator
from sparse_llm.models import (
    DevicePolicy,
    ModelAdapter,
    ModelCapabilities,
    ModelRegistry,
    TransformersCausalLMAdapter,
    UniversalMoEAdapter,
    create_model_adapter,
    get_default_registry,
)
from sparse_llm.inference.engine import InferenceEngine

__all__ = [
    "DevicePolicy",
    "ExpertCache",
    "ExpertKey",
    "GenerationMetrics",
    "GenerationResult",
    "InferenceEngine",
    "ModelAdapter",
    "ModelCapabilities",
    "ModelRegistry",
    "TransformersCausalLMAdapter",
    "UnifiedMemoryCoordinator",
    "UniversalMoEAdapter",
    "create_model_adapter",
    "get_default_registry",
]

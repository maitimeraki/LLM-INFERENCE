from sparse_llm.models.adapters import (
    DevicePolicy,
    ExpertKey,
    ModelAdapter,
    ModelCapabilities,
    TransformersCausalLMAdapter,
)
from sparse_llm.models.paging import (
    LoadedExpertProtocol,
    PagedMoELayer,
    PagingCapabilities,
    PagingValidationResult,
    RouterSelection,
)
from sparse_llm.models.registry import (
    ModelRegistry,
    create_model_adapter,
    get_default_registry,
)
from sparse_llm.models.universal_adapter import UniversalMoEAdapter

__all__ = [
    "DevicePolicy",
    "ExpertKey",
    "LoadedExpertProtocol",
    "ModelAdapter",
    "ModelCapabilities",
    "ModelRegistry",
    "PagedMoELayer",
    "PagingCapabilities",
    "PagingValidationResult",
    "RouterSelection",
    "TransformersCausalLMAdapter",
    "UniversalMoEAdapter",
    "create_model_adapter",
    "get_default_registry",
]

# The legacy toy layer remains importable for compatibility but is not used by
# the production adapter/runtime path.
from sparse_llm.models.moe import RouterGate, SparseMoELayer

__all__ += ["RouterGate", "SparseMoELayer"]

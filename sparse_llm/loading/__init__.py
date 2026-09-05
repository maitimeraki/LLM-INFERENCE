"""Resource-aware weight loading system with four-phase initialization.

Public API:
    FourPhaseOrchestrator: Main entry point for initializing models
    LoadedWeightState: Result container with weights and cache
    ResourceBudget: Hardware capacity information
    PlacementPlan: Weight placement strategy

Usage:
    >>> from sparse_llm.loading import FourPhaseOrchestrator
    >>> orchestrator = FourPhaseOrchestrator()
    >>> state = orchestrator.initialize("mistralai/Mixtral-8x7B-Instruct-v0.1")
    >>> # state.shared_weights, state.expert_cache now ready for inference
"""

from sparse_llm.loading.orchestrator import FourPhaseOrchestrator
from sparse_llm.loading.loaded_weight_state import LoadedWeightState
from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo
from sparse_llm.loading.placement_plan import PlacementPlan
from sparse_llm.loading.model_introspector import ModelInfo
from sparse_llm.loading.expert_cache import ExpertCache

__all__ = [
    "FourPhaseOrchestrator",
    "LoadedWeightState",
    "ResourceBudget",
    "GPUInfo",
    "CPUInfo",
    "StorageInfo",
    "PlacementPlan",
    "ModelInfo",
    "ExpertCache",
]

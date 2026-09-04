"""Four-phase initialization orchestrator for resource-aware weight loading."""

from __future__ import annotations
import time
from typing import Callable

from sparse_llm.loading.resource_profiler import ResourceProfiler
from sparse_llm.loading.model_introspector import ModelIntrospector
from sparse_llm.loading.placement_strategy import PlacementStrategyCalculator
from sparse_llm.loading.weight_loader import WeightLoader
from sparse_llm.loading.loaded_weight_state import LoadedWeightState


class FourPhaseOrchestrator:
    """Orchestrate four-phase initialization: Profile → Plan → Load → Ready."""

    def __init__(self):
        """Initialize orchestrator with default components."""
        self.profiler = ResourceProfiler()
        self.introspector = ModelIntrospector()
        self.calculator = PlacementStrategyCalculator()
        self.loader = WeightLoader()

    def initialize(
        self,
        model_id: str,
        *,
        storage_path: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        **kwargs
    ) -> LoadedWeightState:
        """Execute four-phase initialization.

        Args:
            model_id: Hugging Face model ID or local path
            storage_path: Optional custom storage path for cold experts
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments (reserved for future use)

        Returns:
            LoadedWeightState ready for inference

        Raises:
            ValueError: If model cannot fit in available resources
        """
        def log(message: str):
            if progress_callback:
                progress_callback(message)

        start_time = time.time()

        # Phase 1: Resource Profiling
        log("🔍 [Phase 1/4] Profiling system resources...")
        phase1_start = time.time()

        budget = self.profiler.profile(storage_path=storage_path)

        phase1_time = time.time() - phase1_start
        log(f"   ✓ Resources detected in {phase1_time:.1f}s")
        log(f"   GPU: {budget.total_gpu_bytes / 1024**3:.1f}GB available")
        log(f"   CPU: {budget.total_cpu_bytes / 1024**3:.1f}GB available")
        log(f"   Storage: {budget.storage.path} ({budget.storage.available_bytes / 1024**3:.0f}GB)")

        # Show warnings if any
        for warning in budget.warnings:
            log(f"   ⚠️  {warning}")

        # Phase 2: Placement Strategy
        log("")
        log("📊 [Phase 2/4] Calculating placement strategy...")
        phase2_start = time.time()

        model_info = self.introspector.introspect(model_id)
        plan = self.calculator.calculate(budget, model_info)

        phase2_time = time.time() - phase2_start
        log(f"   ✓ Placement plan calculated in {phase2_time:.1f}s")
        log(f"   Model: {model_info.model_id} ({'MoE' if model_info.is_moe else 'Dense'})")
        log(f"   Total size: {model_info.total_bytes / 1024**3:.1f}GB")

        if model_info.is_moe:
            log(f"   Hot experts (GPU): {plan.hot_expert_slots} slots")
            log(f"   Warm experts (CPU): {plan.warm_expert_slots} slots")
            log(f"   Cold experts (Storage): {plan.cold_expert_count} experts")

        log(f"   GPU utilization: {plan.gpu_utilization_pct:.1f}%")
        log(f"   CPU utilization: {plan.cpu_utilization_pct:.1f}%")

        # Phase 3: Weight Loading
        log("")
        log("⏳ [Phase 3/4] Loading model weights...")
        phase3_start = time.time()

        state = self.loader.load(model_id, plan, progress_callback=log)

        phase3_time = time.time() - phase3_start
        log(f"   ✓ Weights loaded in {phase3_time:.1f}s")

        # Phase 4: Ready
        log("")
        log("✅ [Phase 4/4] Initialization complete")

        total_time = time.time() - start_time
        log(f"   Total time: {total_time:.1f}s")
        log("")
        log("Ready for inference!")

        return state

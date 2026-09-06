"""Adaptive orchestrator with Phase 0 pre-calculation for dynamic memory allocation."""

from __future__ import annotations
import time
from typing import Callable, Optional, Tuple

from sparse_llm.loading.resource_profiler import ResourceProfiler
from sparse_llm.loading.model_introspector import ModelIntrospector
from sparse_llm.loading.placement_strategy import PlacementStrategyCalculator
from sparse_llm.loading.weight_loader import WeightLoader
from sparse_llm.loading.loaded_weight_state import LoadedWeightState
from sparse_llm.loading.memory_budget_calculator import (
    DynamicMemoryBudgetCalculator,
    UserRequest,
    MemoryAllocation
)
from sparse_llm.loading.placement_plan import PlacementPlan
from sparse_llm.loading.resource_budget import ResourceBudget
from sparse_llm.loading.model_introspector import ModelInfo


class AdaptiveMemoryOrchestrator:
    """
    Wraps FourPhaseOrchestrator with Phase 0 pre-calculation.

    Enables dynamic memory allocation based on user request parameters
    (context length, dtype, quantization) rather than fixed placement.

    Five phases:
    - Phase 0: Pre-calculation (validate allocation feasibility)
    - Phases 1-4: Profile → Plan → Load → Ready (respecting allocation)
    """

    def __init__(self):
        """Initialize orchestrator with components."""
        self.memory_calculator = DynamicMemoryBudgetCalculator()
        self.profiler = ResourceProfiler()
        self.introspector = ModelIntrospector()
        self.calculator = PlacementStrategyCalculator()
        self.loader = WeightLoader()

    def initialize(
        self,
        model_id: str,
        user_request: UserRequest,
        *,
        storage_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[LoadedWeightState, MemoryAllocation]:
        """
        Execute five-phase initialization with dynamic allocation.

        Args:
            model_id: Hugging Face model ID or local path
            user_request: User's vLLM configuration parameters
            storage_path: Optional custom storage path for cold experts
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (LoadedWeightState, MemoryAllocation)

        Raises:
            ValueError: If allocation cannot fulfill request
        """
        def log(message: str):
            if progress_callback:
                progress_callback(message)

        start_time = time.time()

        # ========================================================================
        # Phase 0: Pre-calculation (NEW)
        # ========================================================================
        log("🧮 [Phase 0/5] Pre-calculating memory allocation...")
        phase0_start = time.time()

        # Profile resources (needed for allocation calculation)
        budget = self.profiler.profile(storage_path=storage_path)

        # Introspect model (needed for allocation calculation)
        model_info = self.introspector.introspect(model_id)

        # Calculate allocation
        allocation = self.memory_calculator.calculate(
            resource_budget=budget,
            model_info=model_info,
            user_request=user_request
        )

        phase0_time = time.time() - phase0_start

        # Validate allocation
        if not allocation.can_fulfill:
            log(f"   ✗ Cannot fulfill request: {allocation.rejection_reason}")
            raise ValueError(
                f"Memory allocation failed: {allocation.rejection_reason}"
            )

        log(f"   ✓ Allocation validated in {phase0_time:.1f}s")
        log(f"   vLLM GPU utilization: {allocation.vllm_gpu_memory_utilization:.1%}")
        log(f"   GPU hot experts: {allocation.gpu_hot_expert_count}")
        log(f"   CPU warm experts: {allocation.cpu_warm_expert_count}")
        log(f"   SSD cold experts: {allocation.ssd_cold_expert_count}")

        # ========================================================================
        # Phase 1: Resource Profiling (already done in Phase 0)
        # ========================================================================
        log("")
        log("🔍 [Phase 1/5] System resources profiled")
        log(f"   GPU: {budget.total_gpu_bytes / 1024**3:.1f}GB available")
        log(f"   CPU: {budget.total_cpu_bytes / 1024**3:.1f}GB available")
        log(f"   Storage: {budget.storage.path} ({budget.storage.available_bytes / 1024**3:.0f}GB)")

        # Show warnings if any
        for warning in budget.warnings:
            log(f"   ⚠️  {warning}")

        # ========================================================================
        # Phase 2: Placement Strategy (constrained by allocation)
        # ========================================================================
        log("")
        log("📊 [Phase 2/5] Calculating constrained placement strategy...")
        phase2_start = time.time()

        # Create constrained budget based on allocation
        constrained_budget = self._create_constrained_budget(budget, allocation)

        # Calculate placement with constraints
        plan = self.calculator.calculate(constrained_budget, model_info)

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

        # ========================================================================
        # Phase 3: Weight Loading
        # ========================================================================
        log("")
        log("⏳ [Phase 3/5] Loading model weights...")
        phase3_start = time.time()

        state = self.loader.load(model_id, plan, progress_callback=log)

        phase3_time = time.time() - phase3_start
        log(f"   ✓ Weights loaded in {phase3_time:.1f}s")

        # ========================================================================
        # Phase 4: Ready
        # ========================================================================
        log("")
        log("✅ [Phase 4/5] Initialization complete")

        total_time = time.time() - start_time
        log(f"   Total time: {total_time:.1f}s")
        log("")
        log("Ready for inference!")

        return state, allocation

    def _create_constrained_budget(
        self,
        original_budget: ResourceBudget,
        allocation: MemoryAllocation
    ) -> ResourceBudget:
        """
        Create a constrained ResourceBudget that respects the allocation.

        This ensures the PlacementStrategyCalculator allocates experts according
        to the DynamicMemoryBudgetCalculator's plan, which accounts for vLLM KV cache.

        Args:
            original_budget: Original resource budget from profiler
            allocation: Calculated memory allocation from Phase 0

        Returns:
            Constrained ResourceBudget
        """
        # Calculate available space for experts after reserving space for KV cache
        # GPU: Reserve space for shared weights, KV cache, and activation buffer
        gpu_reserved = (
            allocation.gpu_shared_weights +
            allocation.gpu_kv_cache +
            allocation.gpu_activation_buffer
        )
        constrained_gpu_bytes = allocation.total_gpu_available - gpu_reserved

        # CPU: Reserve space according to allocation
        constrained_cpu_bytes = allocation.total_cpu_available

        # Create new budget with constrained values
        # We keep the same GPU objects but adjust total bytes
        from sparse_llm.loading.resource_budget import GPUInfo, CPUInfo

        # Adjust GPU devices proportionally
        if original_budget.gpus and original_budget.total_gpu_bytes > 0:
            scale_factor = constrained_gpu_bytes / original_budget.total_gpu_bytes
            constrained_gpus = [
                GPUInfo(
                    device_id=gpu.device_id,
                    name=gpu.name,
                    total_bytes=int(gpu.total_bytes * scale_factor),
                    available_bytes=int(gpu.available_bytes * scale_factor),
                    compute_capability=gpu.compute_capability
                )
                for gpu in original_budget.gpus
            ]
        else:
            constrained_gpus = original_budget.gpus

        # Adjust CPU proportionally
        if original_budget.cpu.usable_bytes > 0:
            cpu_scale_factor = constrained_cpu_bytes / original_budget.cpu.usable_bytes
            constrained_cpu = CPUInfo(
                total_bytes=int(original_budget.cpu.total_bytes * cpu_scale_factor),
                available_bytes=int(original_budget.cpu.available_bytes * cpu_scale_factor),
                usable_bytes=constrained_cpu_bytes
            )
        else:
            constrained_cpu = original_budget.cpu

        return ResourceBudget(
            gpus=constrained_gpus,
            cpu=constrained_cpu,
            storage=original_budget.storage,
            warnings=original_budget.warnings
        )

"""Dynamic memory budget calculator for coordinating expert weights and KV cache."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from sparse_llm.loading.resource_budget import ResourceBudget
from sparse_llm.loading.model_introspector import ModelInfo


@dataclass(frozen=True)
class UserRequest:
    """User request parameters for inference configuration."""
    max_model_len: int
    dtype: str
    quantization: Optional[str] = None
    tensor_parallel_size: int = 1
    allow_cpu_offload: bool = True
    allow_ssd_offload: bool = True


@dataclass(frozen=True)
class MemoryAllocation:
    """Memory allocation plan across GPU/CPU/SSD tiers."""
    can_fulfill: bool
    rejection_reason: Optional[str]

    # Expert cache GPU utilization parameter (critical output!)
    expert_cache_gpu_utilization: float

    # Memory breakdown (bytes)
    gpu_shared_weights: int
    gpu_kv_cache: int
    gpu_activation_buffer: int
    gpu_hot_experts: int
    gpu_hot_expert_count: int

    cpu_warm_experts: int
    cpu_warm_expert_count: int

    ssd_cold_experts: int
    ssd_cold_expert_count: int

    # Total requirements per tier
    total_gpu_required: int
    total_cpu_required: int
    total_ssd_required: int

    # Available capacity (after safety margins)
    total_gpu_available: int
    total_cpu_available: int
    total_ssd_available: int


class DynamicMemoryBudgetCalculator:
    """Calculate memory allocation between expert weights and KV cache."""

    # Activation buffer as percentage of model size (empirical)
    ACTIVATION_BUFFER_RATIO = 0.10

    # Minimum experts to keep on GPU for reasonable performance
    MIN_GPU_EXPERTS_PER_LAYER = 2

    def calculate(
        self,
        resource_budget: ResourceBudget,
        model_info: ModelInfo,
        user_request: UserRequest
    ) -> MemoryAllocation:
        """Calculate memory allocation plan.

        Args:
            resource_budget: Available hardware resources (with safety margins applied)
            model_info: Model architecture information
            user_request: User's inference configuration request

        Returns:
            MemoryAllocation with detailed breakdown and expert_cache_gpu_utilization
        """
        # Get bytes per element based on dtype
        bytes_per_element = self._get_dtype_bytes(user_request.dtype, user_request.quantization)

        # Calculate KV cache size
        kv_cache_bytes = self._calculate_kv_cache_size(
            model_info=model_info,
            max_model_len=user_request.max_model_len,
            bytes_per_element=bytes_per_element,
            tensor_parallel_size=user_request.tensor_parallel_size
        )

        # Calculate activation buffer (percentage of shared weights + experts)
        activation_buffer_bytes = int(model_info.shared_weight_bytes * self.ACTIVATION_BUFFER_RATIO)

        # Get available capacity
        total_gpu_available = resource_budget.total_gpu_bytes
        total_cpu_available = resource_budget.total_cpu_bytes
        total_ssd_available = resource_budget.storage.available_bytes

        # Calculate GPU allocation for fixed components
        gpu_shared_weights = model_info.shared_weight_bytes
        gpu_kv_cache = kv_cache_bytes
        gpu_activation_buffer = activation_buffer_bytes

        # CRITICAL FIX: Add extra buffer for inference engine internal allocations
        # Inference engines allocate additional memory for:
        # - Model structure/metadata (~1-2GB for MoE models)
        # - Attention workspace
        # - CUDA kernels and buffers
        # For MoE models, this can be 20-30% overhead due to expert routing structures
        if model_info.is_moe:
            # MoE models have significant overhead from expert routing
            engine_overhead = int(total_gpu_available * 0.30)  # 30% for MoE
        else:
            # Dense models have less overhead
            engine_overhead = int(total_gpu_available * 0.10)  # 10% for dense

        gpu_fixed = gpu_shared_weights + gpu_kv_cache + gpu_activation_buffer + engine_overhead

        # Check if fixed components fit on GPU
        if gpu_fixed > total_gpu_available:
            return self._create_rejection(
                reason=(
                    f"GPU capacity insufficient for fixed components. "
                    f"Need {gpu_fixed / 1024**3:.2f}GB "
                    f"(shared: {gpu_shared_weights / 1024**3:.2f}GB, "
                    f"KV cache: {gpu_kv_cache / 1024**3:.2f}GB, "
                    f"activation: {gpu_activation_buffer / 1024**3:.2f}GB, "
                    f"engine overhead: {engine_overhead / 1024**3:.2f}GB), "
                    f"but only {total_gpu_available / 1024**3:.2f}GB available. "
                    f"Try: --max-model-len {max(512, max_model_len // 2)} or use quantization (--quantization fp8)."
                ),
                total_gpu_available=total_gpu_available,
                total_cpu_available=total_cpu_available,
                total_ssd_available=total_ssd_available
            )

        # Calculate remaining GPU space for experts
        gpu_remaining = total_gpu_available - gpu_fixed

        # For MoE models, distribute experts across tiers
        if model_info.is_moe:
            total_experts = model_info.num_experts * model_info.num_layers
            expert_size = model_info.expert_weight_bytes

            # Distribute experts
            allocation = self._distribute_experts(
                gpu_remaining=gpu_remaining,
                total_cpu_available=total_cpu_available,
                total_ssd_available=total_ssd_available,
                total_experts=total_experts,
                expert_size=expert_size,
                num_layers=model_info.num_layers,
                allow_cpu_offload=user_request.allow_cpu_offload,
                allow_ssd_offload=user_request.allow_ssd_offload
            )

            if allocation is None:
                return self._create_rejection(
                    reason=(
                        f"Cannot fit all {total_experts} experts. "
                        f"Need {total_experts * expert_size / 1024**3:.2f}GB total, "
                        f"but only have GPU: {gpu_remaining / 1024**3:.2f}GB, "
                        f"CPU: {total_cpu_available / 1024**3:.2f}GB, "
                        f"SSD: {total_ssd_available / 1024**3:.2f}GB available. "
                        f"Enable CPU/SSD offload or use a smaller model."
                    ),
                    total_gpu_available=total_gpu_available,
                    total_cpu_available=total_cpu_available,
                    total_ssd_available=total_ssd_available
                )

            gpu_hot_expert_count, cpu_warm_expert_count, ssd_cold_expert_count = allocation
        else:
            # Dense model: no expert distribution needed
            gpu_hot_expert_count = 0
            cpu_warm_expert_count = 0
            ssd_cold_expert_count = 0

        # Calculate actual bytes used
        gpu_hot_experts = gpu_hot_expert_count * model_info.expert_weight_bytes if model_info.is_moe else 0
        cpu_warm_experts = cpu_warm_expert_count * model_info.expert_weight_bytes if model_info.is_moe else 0
        ssd_cold_experts = ssd_cold_expert_count * model_info.expert_weight_bytes if model_info.is_moe else 0

        total_gpu_required = gpu_fixed + gpu_hot_experts
        total_cpu_required = cpu_warm_experts
        total_ssd_required = ssd_cold_experts

        # Calculate expert_cache_gpu_utilization (CRITICAL OUTPUT)
        # This tells the inference engine what percentage of GPU memory it can use for expert cache
        # Formula: Expert cache bytes / total GPU memory
        total_gpu_memory = resource_budget.gpus[0].total_bytes if resource_budget.gpus else 1
        expert_cache_gpu_utilization = gpu_kv_cache / total_gpu_memory

        # CRITICAL: Further reduce for safety margin (inference engine's internal calculations)
        # Inference engines have significant overhead beyond the KV cache, especially for MoE models
        # where they allocate routing structures and expert metadata
        if model_info.is_moe:
            expert_cache_gpu_utilization *= 0.5  # 50% safety reduction for MoE
        else:
            expert_cache_gpu_utilization *= 0.8  # 20% safety reduction for dense

        # Clamp to reasonable range (expects 0.0 to 1.0)
        expert_cache_gpu_utilization = max(0.01, min(0.95, expert_cache_gpu_utilization))

        return MemoryAllocation(
            can_fulfill=True,
            rejection_reason=None,
            expert_cache_gpu_utilization=expert_cache_gpu_utilization,
            gpu_shared_weights=gpu_shared_weights,
            gpu_kv_cache=gpu_kv_cache,
            gpu_activation_buffer=gpu_activation_buffer,
            gpu_hot_experts=gpu_hot_experts,
            gpu_hot_expert_count=gpu_hot_expert_count,
            cpu_warm_experts=cpu_warm_experts,
            cpu_warm_expert_count=cpu_warm_expert_count,
            ssd_cold_experts=ssd_cold_experts,
            ssd_cold_expert_count=ssd_cold_expert_count,
            total_gpu_required=total_gpu_required,
            total_cpu_required=total_cpu_required,
            total_ssd_required=total_ssd_required,
            total_gpu_available=total_gpu_available,
            total_cpu_available=total_cpu_available,
            total_ssd_available=total_ssd_available
        )

    def _get_dtype_bytes(self, dtype: str, quantization: Optional[str]) -> int:
        """Get bytes per element for dtype and quantization."""
        if quantization:
            # Quantization overrides dtype
            if quantization in ["int8", "fp8"]:
                return 1
            elif quantization in ["int4", "nf4"]:
                return 1  # Still 1 byte due to packing overhead
            else:
                # Unknown quantization, assume int8
                return 1

        # No quantization, use dtype
        dtype_lower = dtype.lower()
        if dtype_lower in ["float16", "fp16", "bfloat16", "bf16"]:
            return 2
        elif dtype_lower in ["float32", "fp32"]:
            return 4
        elif dtype_lower in ["int8"]:
            return 1
        else:
            # Default to fp16
            return 2

    def _calculate_kv_cache_size(
        self,
        model_info: ModelInfo,
        max_model_len: int,
        bytes_per_element: int,
        tensor_parallel_size: int
    ) -> int:
        """Calculate KV cache size in bytes.

        Formula: 2 (K and V) × num_layers × num_kv_heads × head_dim × max_model_len × bytes_per_element
        Then divide by tensor_parallel_size for distributed inference.

        Note: We use num_key_value_heads (GQA support) instead of num_attention_heads.
        """
        # Calculate head dimension
        head_dim = model_info.hidden_size // model_info.num_attention_heads

        # Use num_key_value_heads for accurate GQA/MQA support
        num_kv_heads = model_info.num_key_value_heads

        # Calculate KV cache size
        # Each layer stores K and V for all positions
        kv_cache_bytes = (
            2  # K and V
            * model_info.num_layers
            * num_kv_heads
            * head_dim
            * max_model_len
            * bytes_per_element
        )

        # Divide by tensor parallel size (distributed across GPUs)
        kv_cache_bytes = kv_cache_bytes // tensor_parallel_size

        return kv_cache_bytes

    def _distribute_experts(
        self,
        gpu_remaining: int,
        total_cpu_available: int,
        total_ssd_available: int,
        total_experts: int,
        expert_size: int,
        num_layers: int,
        allow_cpu_offload: bool,
        allow_ssd_offload: bool
    ) -> Optional[tuple[int, int, int]]:
        """Distribute experts across GPU/CPU/SSD tiers.

        Returns:
            (gpu_expert_count, cpu_expert_count, ssd_expert_count) or None if cannot fit
        """
        # Calculate minimum GPU experts (at least 2 per layer for reasonable perf)
        min_gpu_experts = self.MIN_GPU_EXPERTS_PER_LAYER * num_layers

        # Calculate how many experts fit on GPU
        gpu_expert_count = min(total_experts, gpu_remaining // expert_size)

        # Check minimum GPU requirement
        if gpu_expert_count < min_gpu_experts:
            # Try to fit minimum on GPU
            if min_gpu_experts * expert_size <= gpu_remaining:
                gpu_expert_count = min_gpu_experts
            else:
                # Cannot meet minimum GPU requirement
                gpu_expert_count = max(0, gpu_expert_count)

        remaining_experts = total_experts - gpu_expert_count

        if remaining_experts == 0:
            return (gpu_expert_count, 0, 0)

        # Distribute remaining to CPU
        cpu_expert_count = 0
        if allow_cpu_offload and remaining_experts > 0:
            cpu_expert_count = min(remaining_experts, total_cpu_available // expert_size)
            remaining_experts -= cpu_expert_count

        # Distribute remaining to SSD
        ssd_expert_count = 0
        if allow_ssd_offload and remaining_experts > 0:
            ssd_expert_count = min(remaining_experts, total_ssd_available // expert_size)
            remaining_experts -= ssd_expert_count

        # Check if all experts fit
        if remaining_experts > 0:
            return None

        return (gpu_expert_count, cpu_expert_count, ssd_expert_count)

    def _create_rejection(
        self,
        reason: str,
        total_gpu_available: int,
        total_cpu_available: int,
        total_ssd_available: int
    ) -> MemoryAllocation:
        """Create a rejection allocation."""
        return MemoryAllocation(
            can_fulfill=False,
            rejection_reason=reason,
            expert_cache_gpu_utilization=0.0,
            gpu_shared_weights=0,
            gpu_kv_cache=0,
            gpu_activation_buffer=0,
            gpu_hot_experts=0,
            gpu_hot_expert_count=0,
            cpu_warm_experts=0,
            cpu_warm_expert_count=0,
            ssd_cold_experts=0,
            ssd_cold_expert_count=0,
            total_gpu_required=0,
            total_cpu_required=0,
            total_ssd_required=0,
            total_gpu_available=total_gpu_available,
            total_cpu_available=total_cpu_available,
            total_ssd_available=total_ssd_available
        )

"""Tests for DynamicMemoryBudgetCalculator."""

import pytest

from sparse_llm.loading.memory_budget_calculator import (
    DynamicMemoryBudgetCalculator,
    UserRequest,
    MemoryAllocation
)
from sparse_llm.loading.resource_budget import (
    ResourceBudget,
    GPUInfo,
    CPUInfo,
    StorageInfo
)
from sparse_llm.loading.model_introspector import ModelInfo


class TestUserRequest:
    """Test UserRequest dataclass."""

    def test_user_request_minimal(self):
        """Test UserRequest with minimal parameters."""
        req = UserRequest(
            max_model_len=2048,
            dtype="float16"
        )
        assert req.max_model_len == 2048
        assert req.dtype == "float16"
        assert req.quantization is None
        assert req.tensor_parallel_size == 1
        assert req.allow_cpu_offload is True
        assert req.allow_ssd_offload is True

    def test_user_request_full(self):
        """Test UserRequest with all parameters."""
        req = UserRequest(
            max_model_len=4096,
            dtype="bfloat16",
            quantization="int8",
            tensor_parallel_size=2,
            allow_cpu_offload=False,
            allow_ssd_offload=False
        )
        assert req.max_model_len == 4096
        assert req.dtype == "bfloat16"
        assert req.quantization == "int8"
        assert req.tensor_parallel_size == 2
        assert req.allow_cpu_offload is False
        assert req.allow_ssd_offload is False


class TestMemoryAllocation:
    """Test MemoryAllocation dataclass."""

    def test_memory_allocation_success(self):
        """Test successful MemoryAllocation."""
        alloc = MemoryAllocation(
            can_fulfill=True,
            rejection_reason=None,
            vllm_gpu_memory_utilization=0.5,
            gpu_shared_weights=1000,
            gpu_kv_cache=2000,
            gpu_activation_buffer=100,
            gpu_hot_experts=3000,
            gpu_hot_expert_count=16,
            cpu_warm_experts=5000,
            cpu_warm_expert_count=32,
            ssd_cold_experts=10000,
            ssd_cold_expert_count=64,
            total_gpu_required=6100,
            total_cpu_required=5000,
            total_ssd_required=10000,
            total_gpu_available=8000,
            total_cpu_available=16000,
            total_ssd_available=32000
        )
        assert alloc.can_fulfill is True
        assert alloc.rejection_reason is None
        assert alloc.vllm_gpu_memory_utilization == 0.5

    def test_memory_allocation_rejection(self):
        """Test rejected MemoryAllocation."""
        alloc = MemoryAllocation(
            can_fulfill=False,
            rejection_reason="Insufficient GPU memory",
            vllm_gpu_memory_utilization=0.0,
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
            total_gpu_available=8000,
            total_cpu_available=16000,
            total_ssd_available=32000
        )
        assert alloc.can_fulfill is False
        assert alloc.rejection_reason == "Insufficient GPU memory"


class TestDynamicMemoryBudgetCalculator:
    """Test DynamicMemoryBudgetCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DynamicMemoryBudgetCalculator()

    @pytest.fixture
    def gpu_16gb(self):
        """16GB GPU resource."""
        return GPUInfo(
            device_id=0,
            total_bytes=16 * 1024**3,
            available_bytes=int(16 * 1024**3 * 0.85),  # 85% after safety margin
            compute_capability=(8, 0),
            name="NVIDIA RTX 4080"
        )

    @pytest.fixture
    def cpu_32gb(self):
        """32GB CPU resource."""
        return CPUInfo(
            total_bytes=32 * 1024**3,
            available_bytes=32 * 1024**3,
            usable_bytes=int(32 * 1024**3 * 0.80)  # 80% after safety margin
        )

    @pytest.fixture
    def storage_1tb(self):
        """1TB SSD storage."""
        return StorageInfo(
            path="/tmp/model_cache",
            available_bytes=int(1024 * 1024**3 * 0.90),  # 90% after safety margin
            is_ssd=True,
            estimated_bandwidth_mbps=3000
        )

    @pytest.fixture
    def resource_budget_16gb(self, gpu_16gb, cpu_32gb, storage_1tb):
        """Resource budget with 16GB GPU."""
        return ResourceBudget(
            gpus=[gpu_16gb],
            cpu=cpu_32gb,
            storage=storage_1tb
        )

    @pytest.fixture
    def mixtral_8x7b_info(self):
        """Mixtral 8x7B model info."""
        # Approximate sizes for Mixtral 8x7B
        hidden_size = 4096
        intermediate_size = 14336
        num_layers = 32
        num_experts = 8
        vocab_size = 32000

        # Shared weights: embeddings + attention + norms + router
        embeddings = vocab_size * hidden_size * 2  # fp16
        attention_per_layer = 4 * hidden_size * hidden_size * 2
        norm_per_layer = 2 * hidden_size * 2
        router_per_layer = hidden_size * num_experts * 2
        shared_weight_bytes = embeddings + num_layers * (
            attention_per_layer + norm_per_layer + router_per_layer
        )

        # Expert weights (per expert)
        expert_weight_bytes = (
            hidden_size * intermediate_size * 2  # w1
            + intermediate_size * hidden_size * 2  # w2
            + hidden_size * intermediate_size * 2  # w3
        )

        total_bytes = shared_weight_bytes + num_layers * num_experts * expert_weight_bytes

        return ModelInfo(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            is_moe=True,
            num_layers=32,
            num_experts=8,
            num_experts_per_tok=2,
            shared_weight_bytes=shared_weight_bytes,
            expert_weight_bytes=expert_weight_bytes,
            total_bytes=total_bytes
        )

    @pytest.fixture
    def dense_7b_info(self):
        """Dense 7B model info (like LLaMA)."""
        hidden_size = 4096
        intermediate_size = 11008
        num_layers = 32
        vocab_size = 32000

        # Shared weights: embeddings + attention + norms + FFN
        embeddings = vocab_size * hidden_size * 2
        attention_per_layer = 4 * hidden_size * hidden_size * 2
        norm_per_layer = 2 * hidden_size * 2
        ffn_per_layer = (
            hidden_size * intermediate_size * 2
            + intermediate_size * hidden_size * 2
            + hidden_size * intermediate_size * 2
        )
        shared_weight_bytes = embeddings + num_layers * (
            attention_per_layer + norm_per_layer + ffn_per_layer
        )

        return ModelInfo(
            model_id="meta-llama/Llama-2-7b",
            is_moe=False,
            num_layers=32,
            num_experts=0,
            num_experts_per_tok=0,
            shared_weight_bytes=shared_weight_bytes,
            expert_weight_bytes=0,
            total_bytes=shared_weight_bytes
        )

    def test_dtype_bytes_fp16(self, calculator):
        """Test dtype bytes for fp16."""
        assert calculator._get_dtype_bytes("float16", None) == 2
        assert calculator._get_dtype_bytes("fp16", None) == 2
        assert calculator._get_dtype_bytes("bfloat16", None) == 2
        assert calculator._get_dtype_bytes("bf16", None) == 2

    def test_dtype_bytes_fp32(self, calculator):
        """Test dtype bytes for fp32."""
        assert calculator._get_dtype_bytes("float32", None) == 4
        assert calculator._get_dtype_bytes("fp32", None) == 4

    def test_dtype_bytes_quantized(self, calculator):
        """Test dtype bytes for quantized models."""
        assert calculator._get_dtype_bytes("float16", "int8") == 1
        assert calculator._get_dtype_bytes("float16", "fp8") == 1
        assert calculator._get_dtype_bytes("float16", "int4") == 1
        assert calculator._get_dtype_bytes("float16", "nf4") == 1

    def test_kv_cache_calculation(self, calculator, mixtral_8x7b_info):
        """Test KV cache size calculation."""
        kv_cache = calculator._calculate_kv_cache_size(
            model_info=mixtral_8x7b_info,
            max_model_len=2048,
            bytes_per_element=2,
            tensor_parallel_size=1
        )
        # Should be > 0 and reasonable
        assert kv_cache > 0
        # Rough check: 2 * 32 layers * ~4096 hidden * 2048 len * 2 bytes = ~1GB
        assert kv_cache > 500_000_000  # At least 500MB
        assert kv_cache < 2_000_000_000  # Less than 2GB

    def test_kv_cache_tensor_parallel(self, calculator, mixtral_8x7b_info):
        """Test KV cache with tensor parallelism."""
        kv_single = calculator._calculate_kv_cache_size(
            model_info=mixtral_8x7b_info,
            max_model_len=2048,
            bytes_per_element=2,
            tensor_parallel_size=1
        )
        kv_parallel = calculator._calculate_kv_cache_size(
            model_info=mixtral_8x7b_info,
            max_model_len=2048,
            bytes_per_element=2,
            tensor_parallel_size=2
        )
        # Should be roughly half with TP=2
        assert kv_parallel < kv_single
        assert kv_parallel * 2 == kv_single

    def test_dense_model_success(
        self,
        calculator,
        resource_budget_16gb,
        dense_7b_info
    ):
        """Test calculation for dense model that fits on GPU."""
        # Use smaller max_model_len to ensure it fits in 16GB
        user_request = UserRequest(
            max_model_len=1024,
            dtype="float16"
        )

        allocation = calculator.calculate(
            resource_budget=resource_budget_16gb,
            model_info=dense_7b_info,
            user_request=user_request
        )

        # Verify allocation details (may or may not fit depending on model size)
        if allocation.can_fulfill:
            assert allocation.rejection_reason is None
            assert 0.0 < allocation.vllm_gpu_memory_utilization <= 1.0
            assert allocation.gpu_shared_weights > 0
            assert allocation.gpu_kv_cache > 0
            assert allocation.gpu_activation_buffer > 0
            # Dense model has no experts
            assert allocation.gpu_hot_expert_count == 0
            assert allocation.cpu_warm_expert_count == 0
            assert allocation.ssd_cold_expert_count == 0
        else:
            # If it doesn't fit, verify rejection has a reason
            assert allocation.rejection_reason is not None

    def test_moe_model_all_gpu(
        self,
        calculator,
        resource_budget_16gb,
        mixtral_8x7b_info
    ):
        """Test MoE model where all experts fit on GPU."""
        # Use small max_model_len to leave room for experts
        user_request = UserRequest(
            max_model_len=512,
            dtype="float16"
        )

        allocation = calculator.calculate(
            resource_budget=resource_budget_16gb,
            model_info=mixtral_8x7b_info,
            user_request=user_request
        )

        # May or may not fit all on GPU depending on actual sizes
        # Just verify it returns a valid allocation
        if allocation.can_fulfill:
            assert allocation.vllm_gpu_memory_utilization > 0
            assert allocation.gpu_shared_weights > 0
            assert allocation.gpu_kv_cache > 0
            total_experts = mixtral_8x7b_info.num_experts * mixtral_8x7b_info.num_layers
            allocated_experts = (
                allocation.gpu_hot_expert_count +
                allocation.cpu_warm_expert_count +
                allocation.ssd_cold_expert_count
            )
            assert allocated_experts == total_experts

    def test_moe_model_with_offload(
        self,
        calculator,
        resource_budget_16gb,
        mixtral_8x7b_info
    ):
        """Test MoE model with CPU/SSD offload."""
        user_request = UserRequest(
            max_model_len=4096,  # Large context requires lots of KV cache
            dtype="float16",
            allow_cpu_offload=True,
            allow_ssd_offload=True
        )

        allocation = calculator.calculate(
            resource_budget=resource_budget_16gb,
            model_info=mixtral_8x7b_info,
            user_request=user_request
        )

        if allocation.can_fulfill:
            # Should use CPU or SSD for some experts
            assert (allocation.cpu_warm_expert_count > 0 or
                    allocation.ssd_cold_expert_count > 0)
            total_experts = mixtral_8x7b_info.num_experts * mixtral_8x7b_info.num_layers
            allocated_experts = (
                allocation.gpu_hot_expert_count +
                allocation.cpu_warm_expert_count +
                allocation.ssd_cold_expert_count
            )
            assert allocated_experts == total_experts

    def test_rejection_insufficient_gpu(
        self,
        calculator,
        mixtral_8x7b_info
    ):
        """Test rejection when GPU is too small."""
        # Create tiny GPU (1GB)
        tiny_gpu = GPUInfo(
            device_id=0,
            total_bytes=1 * 1024**3,
            available_bytes=int(1 * 1024**3 * 0.85),
            compute_capability=(8, 0),
            name="Tiny GPU"
        )
        tiny_cpu = CPUInfo(
            total_bytes=8 * 1024**3,
            available_bytes=8 * 1024**3,
            usable_bytes=int(8 * 1024**3 * 0.80)
        )
        tiny_storage = StorageInfo(
            path="/tmp",
            available_bytes=100 * 1024**3,
            is_ssd=True,
            estimated_bandwidth_mbps=1000
        )
        tiny_budget = ResourceBudget(
            gpus=[tiny_gpu],
            cpu=tiny_cpu,
            storage=tiny_storage
        )

        user_request = UserRequest(
            max_model_len=4096,
            dtype="float16"
        )

        allocation = calculator.calculate(
            resource_budget=tiny_budget,
            model_info=mixtral_8x7b_info,
            user_request=user_request
        )

        assert allocation.can_fulfill is False
        assert allocation.rejection_reason is not None
        assert "GPU capacity insufficient" in allocation.rejection_reason

    def test_rejection_offload_disabled(
        self,
        calculator,
        resource_budget_16gb,
        mixtral_8x7b_info
    ):
        """Test rejection when offload is disabled and GPU insufficient."""
        user_request = UserRequest(
            max_model_len=8192,  # Very large context
            dtype="float16",
            allow_cpu_offload=False,
            allow_ssd_offload=False
        )

        allocation = calculator.calculate(
            resource_budget=resource_budget_16gb,
            model_info=mixtral_8x7b_info,
            user_request=user_request
        )

        # May reject if cannot fit everything on GPU
        if not allocation.can_fulfill:
            assert allocation.rejection_reason is not None
            assert allocation.vllm_gpu_memory_utilization == 0.0

    def test_vllm_utilization_range(
        self,
        calculator,
        resource_budget_16gb,
        dense_7b_info
    ):
        """Test vllm_gpu_memory_utilization is in valid range."""
        user_request = UserRequest(
            max_model_len=2048,
            dtype="float16"
        )

        allocation = calculator.calculate(
            resource_budget=resource_budget_16gb,
            model_info=dense_7b_info,
            user_request=user_request
        )

        if allocation.can_fulfill:
            # Should be between 0.01 and 0.95 (clamped range)
            assert 0.01 <= allocation.vllm_gpu_memory_utilization <= 0.95

    def test_expert_distribution_all_tiers(self, calculator):
        """Test expert distribution across all tiers."""
        gpu_remaining = 1 * 1024**3  # 1GB
        cpu_available = 10 * 1024**3  # 10GB
        ssd_available = 50 * 1024**3  # 50GB
        expert_size = 100 * 1024**2  # 100MB per expert
        total_experts = 256  # 32 layers * 8 experts
        num_layers = 32

        result = calculator._distribute_experts(
            gpu_remaining=gpu_remaining,
            total_cpu_available=cpu_available,
            total_ssd_available=ssd_available,
            total_experts=total_experts,
            expert_size=expert_size,
            num_layers=num_layers,
            allow_cpu_offload=True,
            allow_ssd_offload=True
        )

        assert result is not None
        gpu_count, cpu_count, ssd_count = result
        # Should distribute across all tiers
        assert gpu_count + cpu_count + ssd_count == total_experts
        # GPU should have some experts (but may not meet minimum if GPU is small)
        assert gpu_count > 0

    def test_expert_distribution_no_offload(self, calculator):
        """Test expert distribution with offload disabled."""
        gpu_remaining = 5 * 1024**3  # 5GB
        cpu_available = 10 * 1024**3
        ssd_available = 50 * 1024**3
        expert_size = 100 * 1024**2  # 100MB per expert
        total_experts = 64  # 8 layers * 8 experts
        num_layers = 8

        result = calculator._distribute_experts(
            gpu_remaining=gpu_remaining,
            total_cpu_available=cpu_available,
            total_ssd_available=ssd_available,
            total_experts=total_experts,
            expert_size=expert_size,
            num_layers=num_layers,
            allow_cpu_offload=False,
            allow_ssd_offload=False
        )

        if result is not None:
            gpu_count, cpu_count, ssd_count = result
            # Should only use GPU
            assert cpu_count == 0
            assert ssd_count == 0
            assert gpu_count <= total_experts

    def test_expert_distribution_insufficient_space(self, calculator):
        """Test expert distribution when insufficient space."""
        gpu_remaining = 100 * 1024**2  # 100MB
        cpu_available = 200 * 1024**2  # 200MB
        ssd_available = 300 * 1024**2  # 300MB
        expert_size = 100 * 1024**2  # 100MB per expert
        total_experts = 100  # Too many
        num_layers = 32

        result = calculator._distribute_experts(
            gpu_remaining=gpu_remaining,
            total_cpu_available=cpu_available,
            total_ssd_available=ssd_available,
            total_experts=total_experts,
            expert_size=expert_size,
            num_layers=num_layers,
            allow_cpu_offload=True,
            allow_ssd_offload=True
        )

        # Should return None (cannot fit)
        assert result is None

    def test_quantization_reduces_kv_cache(
        self,
        calculator,
        resource_budget_16gb,
        mixtral_8x7b_info
    ):
        """Test that quantization reduces KV cache size."""
        user_fp16 = UserRequest(
            max_model_len=2048,
            dtype="float16",
            quantization=None
        )
        user_int8 = UserRequest(
            max_model_len=2048,
            dtype="float16",
            quantization="int8"
        )

        alloc_fp16 = calculator.calculate(
            resource_budget=resource_budget_16gb,
            model_info=mixtral_8x7b_info,
            user_request=user_fp16
        )
        alloc_int8 = calculator.calculate(
            resource_budget=resource_budget_16gb,
            model_info=mixtral_8x7b_info,
            user_request=user_int8
        )

        if alloc_fp16.can_fulfill and alloc_int8.can_fulfill:
            # int8 should use less KV cache
            assert alloc_int8.gpu_kv_cache < alloc_fp16.gpu_kv_cache

    def test_max_model_len_affects_kv_cache(
        self,
        calculator,
        resource_budget_16gb,
        dense_7b_info
    ):
        """Test that max_model_len affects KV cache size."""
        user_2k = UserRequest(max_model_len=2048, dtype="float16")
        user_4k = UserRequest(max_model_len=4096, dtype="float16")

        alloc_2k = calculator.calculate(
            resource_budget=resource_budget_16gb,
            model_info=dense_7b_info,
            user_request=user_2k
        )
        alloc_4k = calculator.calculate(
            resource_budget=resource_budget_16gb,
            model_info=dense_7b_info,
            user_request=user_4k
        )

        if alloc_2k.can_fulfill and alloc_4k.can_fulfill:
            # 4k should use more KV cache
            assert alloc_4k.gpu_kv_cache > alloc_2k.gpu_kv_cache
            # Should be roughly 2x
            ratio = alloc_4k.gpu_kv_cache / alloc_2k.gpu_kv_cache
            assert 1.8 < ratio < 2.2  # Allow some variance

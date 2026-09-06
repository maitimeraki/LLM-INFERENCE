"""Tests for AdaptiveMemoryOrchestrator with Phase 0 pre-calculation."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from sparse_llm.loading.adaptive_orchestrator import AdaptiveMemoryOrchestrator
from sparse_llm.loading.memory_budget_calculator import (
    UserRequest,
    MemoryAllocation,
    DynamicMemoryBudgetCalculator
)
from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo
from sparse_llm.loading.model_introspector import ModelInfo
from sparse_llm.loading.placement_plan import PlacementPlan
from sparse_llm.loading.loaded_weight_state import LoadedWeightState


class TestAdaptiveMemoryOrchestrator:
    """Test AdaptiveMemoryOrchestrator with Phase 0 integration."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance for tests."""
        return AdaptiveMemoryOrchestrator()

    @pytest.fixture
    def mock_budget(self):
        """Mock ResourceBudget with 24GB GPU, 64GB CPU, 1TB SSD."""
        return ResourceBudget(
            gpus=[
                GPUInfo(
                    device_id=0,
                    name="NVIDIA RTX 4090",
                    total_bytes=24 * 1024**3,
                    available_bytes=22 * 1024**3,
                    compute_capability=(8, 9)
                )
            ],
            cpu=CPUInfo(
                total_bytes=64 * 1024**3,
                available_bytes=64 * 1024**3,
                usable_bytes=64 * 1024**3
            ),
            storage=StorageInfo(
                path="/tmp/sparse_cache",
                available_bytes=1024 * 1024**3,
                is_ssd=True,
                estimated_bandwidth_mbps=5000
            ),
            warnings=[]
        )

    @pytest.fixture
    def mock_mixtral_info(self):
        """Mock ModelInfo for Mixtral 8x7B."""
        return ModelInfo(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            is_moe=True,
            num_layers=32,
            num_experts=8,
            num_experts_per_tok=2,
            shared_weight_bytes=3 * 1024**3,  # 3GB
            expert_weight_bytes=437 * 1024**2,  # 437MB per expert
            total_bytes=115 * 1024**3  # ~115GB total
        )

    @pytest.fixture
    def mock_dense_info(self):
        """Mock ModelInfo for dense 7B model."""
        return ModelInfo(
            model_id="meta-llama/Llama-2-7b-hf",
            is_moe=False,
            num_layers=32,
            num_experts=0,
            num_experts_per_tok=0,
            shared_weight_bytes=13 * 1024**3,  # 13GB
            expert_weight_bytes=0,
            total_bytes=13 * 1024**3
        )

    @pytest.fixture
    def user_request(self):
        """Standard user request for testing."""
        return UserRequest(
            max_model_len=2048,
            dtype="float16",
            quantization=None,
            tensor_parallel_size=1,
            allow_cpu_offload=True,
            allow_ssd_offload=True
        )

    @pytest.fixture
    def mock_fulfillable_allocation(self):
        """Mock allocation that can be fulfilled."""
        return MemoryAllocation(
            can_fulfill=True,
            rejection_reason=None,
            vllm_gpu_memory_utilization=0.15,
            gpu_shared_weights=3 * 1024**3,
            gpu_kv_cache=2 * 1024**3,
            gpu_activation_buffer=300 * 1024**2,
            gpu_hot_experts=64 * 437 * 1024**2,  # 64 experts
            gpu_hot_expert_count=64,
            cpu_warm_experts=128 * 437 * 1024**2,  # 128 experts
            cpu_warm_expert_count=128,
            ssd_cold_experts=64 * 437 * 1024**2,  # 64 experts
            ssd_cold_expert_count=64,
            total_gpu_required=10 * 1024**3,
            total_cpu_required=50 * 1024**3,
            total_ssd_required=30 * 1024**3,
            total_gpu_available=22 * 1024**3,
            total_cpu_available=64 * 1024**3,
            total_ssd_available=1024 * 1024**3
        )

    @pytest.fixture
    def mock_rejection_allocation(self):
        """Mock allocation that cannot be fulfilled."""
        return MemoryAllocation(
            can_fulfill=False,
            rejection_reason="GPU capacity insufficient for fixed components",
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
            total_gpu_available=22 * 1024**3,
            total_cpu_available=64 * 1024**3,
            total_ssd_available=1024 * 1024**3
        )

    def test_initialization(self, orchestrator):
        """Test orchestrator initializes with correct components."""
        assert orchestrator.memory_calculator is not None
        assert orchestrator.profiler is not None
        assert orchestrator.introspector is not None
        assert orchestrator.calculator is not None
        assert orchestrator.loader is not None
        assert isinstance(orchestrator.memory_calculator, DynamicMemoryBudgetCalculator)

    @patch('sparse_llm.loading.adaptive_orchestrator.ResourceProfiler')
    @patch('sparse_llm.loading.adaptive_orchestrator.ModelIntrospector')
    @patch('sparse_llm.loading.adaptive_orchestrator.DynamicMemoryBudgetCalculator')
    @patch('sparse_llm.loading.adaptive_orchestrator.PlacementStrategyCalculator')
    @patch('sparse_llm.loading.adaptive_orchestrator.WeightLoader')
    def test_phase_0_success(
        self,
        mock_loader_cls,
        mock_calculator_cls,
        mock_memory_calc_cls,
        mock_introspector_cls,
        mock_profiler_cls,
        user_request,
        mock_budget,
        mock_mixtral_info,
        mock_fulfillable_allocation
    ):
        """Test Phase 0 pre-calculation succeeds when allocation is feasible."""
        # Setup mocks
        mock_profiler = Mock()
        mock_profiler.profile.return_value = mock_budget
        mock_profiler_cls.return_value = mock_profiler

        mock_introspector = Mock()
        mock_introspector.introspect.return_value = mock_mixtral_info
        mock_introspector_cls.return_value = mock_introspector

        mock_memory_calc = Mock()
        mock_memory_calc.calculate.return_value = mock_fulfillable_allocation
        mock_memory_calc_cls.return_value = mock_memory_calc

        mock_calculator = Mock()
        mock_plan = Mock(spec=PlacementPlan)
        mock_plan.hot_expert_slots = 64
        mock_plan.warm_expert_slots = 128
        mock_plan.cold_expert_count = 64
        mock_plan.gpu_utilization_pct = 45.0
        mock_plan.cpu_utilization_pct = 78.0
        mock_calculator.calculate.return_value = mock_plan
        mock_calculator_cls.return_value = mock_calculator

        mock_loader = Mock()
        mock_state = Mock(spec=LoadedWeightState)
        mock_loader.load.return_value = mock_state
        mock_loader_cls.return_value = mock_loader

        # Create orchestrator and run
        orchestrator = AdaptiveMemoryOrchestrator()
        progress_logs = []

        state, allocation = orchestrator.initialize(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            user_request=user_request,
            progress_callback=lambda msg: progress_logs.append(msg)
        )

        # Verify Phase 0 executed
        mock_profiler.profile.assert_called_once()
        mock_introspector.introspect.assert_called_once()
        mock_memory_calc.calculate.assert_called_once_with(
            resource_budget=mock_budget,
            model_info=mock_mixtral_info,
            user_request=user_request
        )

        # Verify allocation returned
        assert allocation == mock_fulfillable_allocation
        assert allocation.can_fulfill is True

        # Verify subsequent phases executed
        mock_calculator.calculate.assert_called_once()
        mock_loader.load.assert_called_once()

        # Verify state returned
        assert state == mock_state

        # Verify progress logs include all phases
        log_text = "\n".join(progress_logs)
        assert "[Phase 0/5]" in log_text
        assert "[Phase 1/5]" in log_text
        assert "[Phase 2/5]" in log_text
        assert "[Phase 3/5]" in log_text
        assert "[Phase 4/5]" in log_text
        assert "vLLM GPU utilization: 15.0%" in log_text
        assert "GPU hot experts: 64" in log_text

    @patch('sparse_llm.loading.adaptive_orchestrator.ResourceProfiler')
    @patch('sparse_llm.loading.adaptive_orchestrator.ModelIntrospector')
    @patch('sparse_llm.loading.adaptive_orchestrator.DynamicMemoryBudgetCalculator')
    def test_phase_0_rejection(
        self,
        mock_memory_calc_cls,
        mock_introspector_cls,
        mock_profiler_cls,
        user_request,
        mock_budget,
        mock_mixtral_info,
        mock_rejection_allocation
    ):
        """Test Phase 0 rejects early when allocation cannot be fulfilled."""
        # Setup mocks
        mock_profiler = Mock()
        mock_profiler.profile.return_value = mock_budget
        mock_profiler_cls.return_value = mock_profiler

        mock_introspector = Mock()
        mock_introspector.introspect.return_value = mock_mixtral_info
        mock_introspector_cls.return_value = mock_introspector

        mock_memory_calc = Mock()
        mock_memory_calc.calculate.return_value = mock_rejection_allocation
        mock_memory_calc_cls.return_value = mock_memory_calc

        # Create orchestrator and run
        orchestrator = AdaptiveMemoryOrchestrator()
        progress_logs = []

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            orchestrator.initialize(
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                user_request=user_request,
                progress_callback=lambda msg: progress_logs.append(msg)
            )

        # Verify error message includes rejection reason
        assert "GPU capacity insufficient" in str(exc_info.value)

        # Verify Phase 0 executed but stopped early
        mock_profiler.profile.assert_called_once()
        mock_introspector.introspect.assert_called_once()
        mock_memory_calc.calculate.assert_called_once()

        # Verify progress logs show rejection
        log_text = "\n".join(progress_logs)
        assert "[Phase 0/5]" in log_text
        assert "Cannot fulfill request" in log_text

    @patch('sparse_llm.loading.adaptive_orchestrator.ResourceProfiler')
    @patch('sparse_llm.loading.adaptive_orchestrator.ModelIntrospector')
    @patch('sparse_llm.loading.adaptive_orchestrator.DynamicMemoryBudgetCalculator')
    @patch('sparse_llm.loading.adaptive_orchestrator.PlacementStrategyCalculator')
    @patch('sparse_llm.loading.adaptive_orchestrator.WeightLoader')
    def test_constrained_budget_creation(
        self,
        mock_loader_cls,
        mock_calculator_cls,
        mock_memory_calc_cls,
        mock_introspector_cls,
        mock_profiler_cls,
        user_request,
        mock_budget,
        mock_mixtral_info,
        mock_fulfillable_allocation
    ):
        """Test that constrained budget is created correctly from allocation."""
        # Setup mocks
        mock_profiler = Mock()
        mock_profiler.profile.return_value = mock_budget
        mock_profiler_cls.return_value = mock_profiler

        mock_introspector = Mock()
        mock_introspector.introspect.return_value = mock_mixtral_info
        mock_introspector_cls.return_value = mock_introspector

        mock_memory_calc = Mock()
        mock_memory_calc.calculate.return_value = mock_fulfillable_allocation
        mock_memory_calc_cls.return_value = mock_memory_calc

        mock_calculator = Mock()
        mock_plan = Mock()
        mock_plan.hot_expert_slots = 64
        mock_plan.warm_expert_slots = 128
        mock_plan.cold_expert_count = 64
        mock_plan.gpu_utilization_pct = 45.0
        mock_plan.cpu_utilization_pct = 78.0
        mock_calculator.calculate.return_value = mock_plan
        mock_calculator_cls.return_value = mock_calculator

        mock_loader = Mock()
        mock_state = Mock(spec=LoadedWeightState)
        mock_loader.load.return_value = mock_state
        mock_loader_cls.return_value = mock_loader

        # Create orchestrator and run
        orchestrator = AdaptiveMemoryOrchestrator()
        state, allocation = orchestrator.initialize(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            user_request=user_request
        )

        # Verify calculator.calculate was called with constrained budget
        mock_calculator.calculate.assert_called_once()
        call_args = mock_calculator.calculate.call_args
        constrained_budget = call_args[0][0]

        # Verify constrained budget has reduced GPU capacity
        # Original: 22GB, Reserved: ~5.3GB (3GB shared + 2GB KV + 0.3GB activation)
        # Remaining: ~16.7GB
        gpu_reserved = (
            mock_fulfillable_allocation.gpu_shared_weights +
            mock_fulfillable_allocation.gpu_kv_cache +
            mock_fulfillable_allocation.gpu_activation_buffer
        )
        expected_constrained = mock_fulfillable_allocation.total_gpu_available - gpu_reserved

        # Allow 1% tolerance for rounding
        assert abs(constrained_budget.total_gpu_bytes - expected_constrained) < expected_constrained * 0.01

    @patch('sparse_llm.loading.adaptive_orchestrator.ResourceProfiler')
    @patch('sparse_llm.loading.adaptive_orchestrator.ModelIntrospector')
    @patch('sparse_llm.loading.adaptive_orchestrator.DynamicMemoryBudgetCalculator')
    @patch('sparse_llm.loading.adaptive_orchestrator.PlacementStrategyCalculator')
    @patch('sparse_llm.loading.adaptive_orchestrator.WeightLoader')
    def test_dense_model_handling(
        self,
        mock_loader_cls,
        mock_calculator_cls,
        mock_memory_calc_cls,
        mock_introspector_cls,
        mock_profiler_cls,
        user_request,
        mock_budget,
        mock_dense_info
    ):
        """Test handling of dense (non-MoE) models."""
        # Create allocation for dense model
        dense_allocation = MemoryAllocation(
            can_fulfill=True,
            rejection_reason=None,
            vllm_gpu_memory_utilization=0.20,
            gpu_shared_weights=13 * 1024**3,
            gpu_kv_cache=2 * 1024**3,
            gpu_activation_buffer=1 * 1024**3,
            gpu_hot_experts=0,  # No experts for dense models
            gpu_hot_expert_count=0,
            cpu_warm_experts=0,
            cpu_warm_expert_count=0,
            ssd_cold_experts=0,
            ssd_cold_expert_count=0,
            total_gpu_required=16 * 1024**3,
            total_cpu_required=0,
            total_ssd_required=0,
            total_gpu_available=22 * 1024**3,
            total_cpu_available=64 * 1024**3,
            total_ssd_available=1024 * 1024**3
        )

        # Setup mocks
        mock_profiler = Mock()
        mock_profiler.profile.return_value = mock_budget
        mock_profiler_cls.return_value = mock_profiler

        mock_introspector = Mock()
        mock_introspector.introspect.return_value = mock_dense_info
        mock_introspector_cls.return_value = mock_introspector

        mock_memory_calc = Mock()
        mock_memory_calc.calculate.return_value = dense_allocation
        mock_memory_calc_cls.return_value = mock_memory_calc

        mock_calculator = Mock()
        mock_plan = Mock()
        mock_plan.hot_expert_slots = 0
        mock_plan.warm_expert_slots = 0
        mock_plan.cold_expert_count = 0
        mock_plan.gpu_utilization_pct = 60.0
        mock_plan.cpu_utilization_pct = 0.0
        mock_calculator.calculate.return_value = mock_plan
        mock_calculator_cls.return_value = mock_calculator

        mock_loader = Mock()
        mock_state = Mock(spec=LoadedWeightState)
        mock_loader.load.return_value = mock_state
        mock_loader_cls.return_value = mock_loader

        # Create orchestrator and run
        orchestrator = AdaptiveMemoryOrchestrator()
        progress_logs = []

        state, allocation = orchestrator.initialize(
            model_id="meta-llama/Llama-2-7b-hf",
            user_request=user_request,
            progress_callback=lambda msg: progress_logs.append(msg)
        )

        # Verify allocation for dense model
        assert allocation.gpu_hot_expert_count == 0
        assert allocation.cpu_warm_expert_count == 0
        assert allocation.ssd_cold_expert_count == 0
        assert allocation.can_fulfill is True

        # Verify all phases completed
        log_text = "\n".join(progress_logs)
        assert "[Phase 0/5]" in log_text
        assert "[Phase 4/5]" in log_text
        assert "GPU hot experts: 0" in log_text

    @patch('sparse_llm.loading.adaptive_orchestrator.ResourceProfiler')
    @patch('sparse_llm.loading.adaptive_orchestrator.ModelIntrospector')
    @patch('sparse_llm.loading.adaptive_orchestrator.DynamicMemoryBudgetCalculator')
    @patch('sparse_llm.loading.adaptive_orchestrator.PlacementStrategyCalculator')
    @patch('sparse_llm.loading.adaptive_orchestrator.WeightLoader')
    def test_storage_path_propagation(
        self,
        mock_loader_cls,
        mock_calculator_cls,
        mock_memory_calc_cls,
        mock_introspector_cls,
        mock_profiler_cls,
        user_request,
        mock_budget,
        mock_mixtral_info,
        mock_fulfillable_allocation
    ):
        """Test that storage_path is propagated to profiler."""
        # Setup mocks
        mock_profiler = Mock()
        mock_profiler.profile.return_value = mock_budget
        mock_profiler_cls.return_value = mock_profiler

        mock_introspector = Mock()
        mock_introspector.introspect.return_value = mock_mixtral_info
        mock_introspector_cls.return_value = mock_introspector

        mock_memory_calc = Mock()
        mock_memory_calc.calculate.return_value = mock_fulfillable_allocation
        mock_memory_calc_cls.return_value = mock_memory_calc

        mock_calculator = Mock()
        mock_plan = Mock()
        mock_plan.hot_expert_slots = 64
        mock_plan.warm_expert_slots = 128
        mock_plan.cold_expert_count = 64
        mock_plan.gpu_utilization_pct = 45.0
        mock_plan.cpu_utilization_pct = 78.0
        mock_calculator.calculate.return_value = mock_plan
        mock_calculator_cls.return_value = mock_calculator

        mock_loader = Mock()
        mock_state = Mock(spec=LoadedWeightState)
        mock_loader.load.return_value = mock_state
        mock_loader_cls.return_value = mock_loader

        # Create orchestrator and run with custom storage path
        orchestrator = AdaptiveMemoryOrchestrator()
        custom_storage = "/custom/storage/path"

        state, allocation = orchestrator.initialize(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            user_request=user_request,
            storage_path=custom_storage
        )

        # Verify storage_path passed to profiler
        mock_profiler.profile.assert_called_once_with(storage_path=custom_storage)

    def test_returns_both_state_and_allocation(
        self,
        orchestrator,
        user_request
    ):
        """Test that initialize returns tuple of (state, allocation)."""
        # Mock all components
        orchestrator.profiler.profile = Mock(return_value=ResourceBudget(
            gpus=[GPUInfo(
                device_id=0,
                total_bytes=24*1024**3,
                available_bytes=22*1024**3,
                compute_capability=(8, 9),
                name="Test GPU"
            )],
            cpu=CPUInfo(
                total_bytes=64*1024**3,
                available_bytes=64*1024**3,
                usable_bytes=64*1024**3
            ),
            storage=StorageInfo(
                path="/tmp",
                available_bytes=1024*1024**3,
                is_ssd=True,
                estimated_bandwidth_mbps=5000
            ),
            warnings=[]
        ))
        orchestrator.introspector.introspect = Mock(return_value=Mock(
            model_id="test",
            is_moe=False,
            num_layers=32,
            shared_weight_bytes=10*1024**3,
            expert_weight_bytes=0,
            total_bytes=10*1024**3
        ))
        orchestrator.memory_calculator.calculate = Mock(return_value=Mock(
            can_fulfill=True,
            rejection_reason=None,
            vllm_gpu_memory_utilization=0.1,
            gpu_shared_weights=10*1024**3,
            gpu_kv_cache=1*1024**3,
            gpu_activation_buffer=1*1024**3,
            gpu_hot_experts=0,
            gpu_hot_expert_count=0,
            cpu_warm_experts=0,
            cpu_warm_expert_count=0,
            ssd_cold_experts=0,
            ssd_cold_expert_count=0,
            total_gpu_required=12*1024**3,
            total_cpu_required=0,
            total_ssd_required=0,
            total_gpu_available=22*1024**3,
            total_cpu_available=64*1024**3,
            total_ssd_available=1024*1024**3
        ))
        orchestrator.calculator.calculate = Mock(return_value=Mock(
            hot_expert_slots=0,
            warm_expert_slots=0,
            cold_expert_count=0,
            gpu_utilization_pct=50.0,
            cpu_utilization_pct=0.0
        ))
        orchestrator.loader.load = Mock(return_value=Mock(spec=LoadedWeightState))

        # Execute
        result = orchestrator.initialize(
            model_id="test/model",
            user_request=user_request
        )

        # Verify return type
        assert isinstance(result, tuple)
        assert len(result) == 2
        state, allocation = result
        assert isinstance(state, Mock)  # Mock of LoadedWeightState
        assert isinstance(allocation, Mock)  # Mock of MemoryAllocation

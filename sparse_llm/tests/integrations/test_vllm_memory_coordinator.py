"""Tests for VLLMMemoryCoordinator."""

import sys
import unittest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any

# Mock vllm module before any imports
sys.modules['vllm'] = MagicMock()

from sparse_llm.integrations.vllm_memory_coordinator import (
    VLLMMemoryCoordinator,
    initialize_vllm_with_coordination
)
from sparse_llm.loading.memory_budget_calculator import (
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
from sparse_llm.loading.loaded_weight_state import LoadedWeightState


class TestVLLMMemoryCoordinator(unittest.TestCase):
    """Test VLLMMemoryCoordinator initialization flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_id = "mistralai/Mixtral-8x7B-v0.1"
        self.user_vllm_params = {
            "max_model_len": 2048,
            "dtype": "float16",
            "quantization": None,
            "tensor_parallel_size": 1
        }

        # Mock resource budget (4GB GPU, 16GB CPU, 100GB SSD)
        self.mock_resource_budget = ResourceBudget(
            gpus=[GPUInfo(
                device_id=0,
                total_bytes=4 * 1024**3,
                available_bytes=int(4 * 1024**3 * 0.85),
                compute_capability=(8, 6),
                name="NVIDIA RTX 3060"
            )],
            cpu=CPUInfo(
                total_bytes=16 * 1024**3,
                available_bytes=12 * 1024**3,
                usable_bytes=int(12 * 1024**3 * 0.80)
            ),
            storage=StorageInfo(
                path="/tmp/sparse_llm",
                available_bytes=100 * 1024**3,
                is_ssd=True,
                estimated_bandwidth_mbps=1200
            )
        )

        # Mock model info (Mixtral-8x7B)
        self.mock_model_info = ModelInfo(
            model_id=self.model_id,
            is_moe=True,
            num_layers=32,
            num_experts=8,
            num_experts_per_tok=2,
            shared_weight_bytes=int(2.5 * 1024**3),  # 2.5GB shared
            expert_weight_bytes=int(0.5 * 1024**3),  # 0.5GB per expert
            total_bytes=int(90 * 1024**3)  # ~90GB total
        )

        # Mock successful allocation
        self.mock_allocation = MemoryAllocation(
            can_fulfill=True,
            rejection_reason=None,
            vllm_gpu_memory_utilization=0.25,  # 25% GPU for KV cache
            gpu_shared_weights=int(2.5 * 1024**3),
            gpu_kv_cache=int(0.8 * 1024**3),
            gpu_activation_buffer=int(0.25 * 1024**3),
            gpu_hot_experts=int(0.5 * 1024**3),  # 1 expert
            gpu_hot_expert_count=1,
            cpu_warm_experts=int(16 * 1024**3),  # 32 experts
            cpu_warm_expert_count=32,
            ssd_cold_experts=int(100 * 1024**3),  # Rest on SSD
            ssd_cold_expert_count=223,
            total_gpu_required=int(4.05 * 1024**3),
            total_cpu_required=int(16 * 1024**3),
            total_ssd_required=int(100 * 1024**3),
            total_gpu_available=int(4 * 1024**3 * 0.85),
            total_cpu_available=int(12 * 1024**3 * 0.80),
            total_ssd_available=100 * 1024**3
        )

    def test_init(self):
        """Test coordinator initialization."""
        coordinator = VLLMMemoryCoordinator(
            model_id=self.model_id,
            user_vllm_params=self.user_vllm_params,
            storage_path="/tmp/test"
        )

        self.assertEqual(coordinator.model_id, self.model_id)
        self.assertEqual(coordinator.user_vllm_params, self.user_vllm_params)
        self.assertEqual(coordinator.storage_path, "/tmp/test")
        self.assertIsNotNone(coordinator.profiler)
        self.assertIsNotNone(coordinator.introspector)
        self.assertIsNotNone(coordinator.calculator)
        self.assertIsNotNone(coordinator.orchestrator)

    def test_init_with_defaults(self):
        """Test coordinator initialization with default parameters."""
        coordinator = VLLMMemoryCoordinator(model_id=self.model_id)

        self.assertEqual(coordinator.model_id, self.model_id)
        self.assertEqual(coordinator.user_vllm_params, {})
        self.assertIsNone(coordinator.storage_path)

    def test_create_user_request(self):
        """Test UserRequest creation from vLLM parameters."""
        coordinator = VLLMMemoryCoordinator(
            model_id=self.model_id,
            user_vllm_params=self.user_vllm_params
        )

        request = coordinator._create_user_request()

        self.assertIsInstance(request, UserRequest)
        self.assertEqual(request.max_model_len, 2048)
        self.assertEqual(request.dtype, "float16")
        self.assertIsNone(request.quantization)
        self.assertEqual(request.tensor_parallel_size, 1)
        self.assertTrue(request.allow_cpu_offload)
        self.assertTrue(request.allow_ssd_offload)

    def test_create_user_request_with_quantization(self):
        """Test UserRequest creation with quantization."""
        params = {
            "max_model_len": 4096,
            "dtype": "float16",
            "quantization": "int8",
            "tensor_parallel_size": 2,
            "allow_cpu_offload": False,
            "allow_ssd_offload": False
        }
        coordinator = VLLMMemoryCoordinator(
            model_id=self.model_id,
            user_vllm_params=params
        )

        request = coordinator._create_user_request()

        self.assertEqual(request.max_model_len, 4096)
        self.assertEqual(request.quantization, "int8")
        self.assertEqual(request.tensor_parallel_size, 2)
        self.assertFalse(request.allow_cpu_offload)
        self.assertFalse(request.allow_ssd_offload)

    def test_create_user_request_defaults(self):
        """Test UserRequest creation with default values."""
        coordinator = VLLMMemoryCoordinator(
            model_id=self.model_id,
            user_vllm_params={}
        )

        request = coordinator._create_user_request()

        self.assertEqual(request.max_model_len, 2048)  # Default
        self.assertEqual(request.dtype, "float16")  # Default
        self.assertIsNone(request.quantization)
        self.assertEqual(request.tensor_parallel_size, 1)

    def test_initialize_vllm_success(self):
        """Test successful vLLM initialization."""
        with patch('vllm.LLM') as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            coordinator = VLLMMemoryCoordinator(
                model_id=self.model_id,
                user_vllm_params=self.user_vllm_params
            )

            engine = coordinator._initialize_vllm(self.mock_allocation)

            # Verify vLLM was initialized with correct parameters
            mock_llm_class.assert_called_once()
            call_kwargs = mock_llm_class.call_args[1]

            self.assertEqual(call_kwargs["model"], self.model_id)
            self.assertEqual(call_kwargs["gpu_memory_utilization"], 0.25)
            self.assertEqual(call_kwargs["max_model_len"], 2048)
            self.assertEqual(call_kwargs["dtype"], "float16")

            self.assertEqual(engine, mock_llm_instance)

    def test_initialize_vllm_overrides_gpu_memory_utilization(self):
        """Test that coordinator overrides user-provided gpu_memory_utilization."""
        with patch('vllm.LLM') as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            # User provides gpu_memory_utilization (should be overridden)
            params_with_gpu_mem = dict(self.user_vllm_params)
            params_with_gpu_mem["gpu_memory_utilization"] = 0.90  # User wants 90%

            coordinator = VLLMMemoryCoordinator(
                model_id=self.model_id,
                user_vllm_params=params_with_gpu_mem
            )

            engine = coordinator._initialize_vllm(self.mock_allocation)

            # Verify coordinator's calculated value is used (0.25), not user's (0.90)
            call_kwargs = mock_llm_class.call_args[1]
            self.assertEqual(call_kwargs["gpu_memory_utilization"], 0.25)

    def test_initialize_vllm_failure(self):
        """Test vLLM initialization failure handling."""
        with patch('vllm.LLM') as mock_llm_class:
            mock_llm_class.side_effect = RuntimeError("CUDA out of memory")

            coordinator = VLLMMemoryCoordinator(
                model_id=self.model_id,
                user_vllm_params=self.user_vllm_params
            )

            with self.assertRaises(RuntimeError) as ctx:
                coordinator._initialize_vllm(self.mock_allocation)

            self.assertIn("Failed to initialize vLLM", str(ctx.exception))
            self.assertIn("CUDA out of memory", str(ctx.exception))

    @patch.object(VLLMMemoryCoordinator, '_initialize_vllm')
    @patch('sparse_llm.integrations.vllm_memory_coordinator.FourPhaseOrchestrator')
    @patch('sparse_llm.integrations.vllm_memory_coordinator.DynamicMemoryBudgetCalculator')
    @patch('sparse_llm.integrations.vllm_memory_coordinator.ModelIntrospector')
    @patch('sparse_llm.integrations.vllm_memory_coordinator.ResourceProfiler')
    def test_initialize_with_coordination_success(
        self,
        mock_profiler_class,
        mock_introspector_class,
        mock_calculator_class,
        mock_orchestrator_class,
        mock_initialize_vllm
    ):
        """Test full initialization flow with successful coordination."""
        # Setup mocks
        mock_profiler = Mock()
        mock_profiler.profile.return_value = self.mock_resource_budget
        mock_profiler_class.return_value = mock_profiler

        mock_introspector = Mock()
        mock_introspector.introspect.return_value = self.mock_model_info
        mock_introspector_class.return_value = mock_introspector

        mock_calculator = Mock()
        mock_calculator.calculate.return_value = self.mock_allocation
        mock_calculator_class.return_value = mock_calculator

        mock_loaded_state = Mock(spec=LoadedWeightState)
        mock_loaded_state.model_info = self.mock_model_info
        mock_orchestrator = Mock()
        mock_orchestrator.initialize.return_value = mock_loaded_state
        mock_orchestrator_class.return_value = mock_orchestrator

        mock_vllm_engine = MagicMock()
        mock_initialize_vllm.return_value = mock_vllm_engine

        # Execute
        coordinator = VLLMMemoryCoordinator(
            model_id=self.model_id,
            user_vllm_params=self.user_vllm_params,
            storage_path="/tmp/test"
        )

        engine, allocation, loaded_state = coordinator.initialize_with_coordination()

        # Verify flow
        mock_profiler.profile.assert_called_once_with(storage_path="/tmp/test")
        mock_introspector.introspect.assert_called_once_with(self.model_id)
        mock_calculator.calculate.assert_called_once()
        mock_orchestrator.initialize.assert_called_once()
        mock_initialize_vllm.assert_called_once_with(self.mock_allocation)

        # Verify returns
        self.assertEqual(engine, mock_vllm_engine)
        self.assertEqual(allocation, self.mock_allocation)
        self.assertEqual(loaded_state, mock_loaded_state)

    @patch('sparse_llm.integrations.vllm_memory_coordinator.DynamicMemoryBudgetCalculator')
    @patch('sparse_llm.integrations.vllm_memory_coordinator.ModelIntrospector')
    @patch('sparse_llm.integrations.vllm_memory_coordinator.ResourceProfiler')
    def test_initialize_with_coordination_rejection(
        self,
        mock_profiler_class,
        mock_introspector_class,
        mock_calculator_class
    ):
        """Test initialization flow when allocation is rejected."""
        # Setup mocks
        mock_profiler = Mock()
        mock_profiler.profile.return_value = self.mock_resource_budget
        mock_profiler_class.return_value = mock_profiler

        mock_introspector = Mock()
        mock_introspector.introspect.return_value = self.mock_model_info
        mock_introspector_class.return_value = mock_introspector

        # Create rejection allocation
        rejection_allocation = MemoryAllocation(
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
            total_gpu_available=int(4 * 1024**3 * 0.85),
            total_cpu_available=int(12 * 1024**3 * 0.80),
            total_ssd_available=100 * 1024**3
        )

        mock_calculator = Mock()
        mock_calculator.calculate.return_value = rejection_allocation
        mock_calculator_class.return_value = mock_calculator

        # Execute and expect ValueError
        coordinator = VLLMMemoryCoordinator(
            model_id=self.model_id,
            user_vllm_params=self.user_vllm_params
        )

        with self.assertRaises(ValueError) as ctx:
            coordinator.initialize_with_coordination()

        self.assertIn("Cannot fulfill memory allocation request", str(ctx.exception))
        self.assertIn("GPU capacity insufficient", str(ctx.exception))

    def test_log_with_callback(self):
        """Test logging with progress callback."""
        messages = []

        def callback(msg):
            messages.append(msg)

        coordinator = VLLMMemoryCoordinator(
            model_id=self.model_id,
            progress_callback=callback
        )

        coordinator._log("Test message 1")
        coordinator._log("Test message 2")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], "Test message 1")
        self.assertEqual(messages[1], "Test message 2")

    def test_log_without_callback(self):
        """Test logging without progress callback (uses logger)."""
        coordinator = VLLMMemoryCoordinator(model_id=self.model_id)

        # Should not raise exception
        coordinator._log("Test message")


class TestConvenienceFunction(unittest.TestCase):
    """Test initialize_vllm_with_coordination convenience function."""

    @patch('sparse_llm.integrations.vllm_memory_coordinator.VLLMMemoryCoordinator')
    def test_initialize_vllm_with_coordination(self, mock_coordinator_class):
        """Test convenience function delegates to coordinator."""
        mock_coordinator = Mock()
        mock_engine = MagicMock()
        mock_allocation = Mock()
        mock_state = Mock()
        mock_coordinator.initialize_with_coordination.return_value = (
            mock_engine, mock_allocation, mock_state
        )
        mock_coordinator_class.return_value = mock_coordinator

        # Call convenience function
        engine, allocation, state = initialize_vllm_with_coordination(
            model_id="test/model",
            max_model_len=4096,
            dtype="float16",
            quantization="int8",
            tensor_parallel_size=2,
            storage_path="/tmp/test",
            trust_remote_code=True
        )

        # Verify coordinator was created with correct parameters
        mock_coordinator_class.assert_called_once()
        call_kwargs = mock_coordinator_class.call_args[1]

        self.assertEqual(call_kwargs["model_id"], "test/model")
        self.assertEqual(call_kwargs["storage_path"], "/tmp/test")

        vllm_params = call_kwargs["user_vllm_params"]
        self.assertEqual(vllm_params["max_model_len"], 4096)
        self.assertEqual(vllm_params["dtype"], "float16")
        self.assertEqual(vllm_params["quantization"], "int8")
        self.assertEqual(vllm_params["tensor_parallel_size"], 2)
        self.assertTrue(vllm_params["trust_remote_code"])

        # Verify returns
        self.assertEqual(engine, mock_engine)
        self.assertEqual(allocation, mock_allocation)
        self.assertEqual(state, mock_state)


if __name__ == "__main__":
    unittest.main()

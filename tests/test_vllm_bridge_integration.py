"""Tests for vLLM bridge integration with memory coordination.

Tests cover the new initialize_vllm_with_coordination() convenience function
that provides OOM-safe vLLM initialization.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any, Dict

from sparse_llm.integrations.vllm_bridge import (
    VLLMWeightBridge,
    initialize_vllm_with_coordination
)


class TestInitializeVLLMWithCoordination:
    """Test the convenience function for coordinated vLLM initialization."""

    @pytest.fixture
    def mock_coordinator(self):
        """Mock VLLMMemoryCoordinator."""
        with patch('sparse_llm.integrations.vllm_memory_coordinator.VLLMMemoryCoordinator') as mock_cls:
            coordinator_instance = Mock()
            mock_cls.return_value = coordinator_instance

            # Mock the return values
            mock_vllm_engine = Mock()

            # Mock vLLM model with named_parameters that returns empty list
            mock_vllm_model = Mock()
            mock_vllm_model.named_parameters.return_value = []
            mock_vllm_model.named_modules.return_value = []

            mock_vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model = mock_vllm_model

            mock_allocation = Mock()
            mock_allocation.gpu_kv_cache = 4 * 1024**3  # 4GB
            mock_allocation.gpu_hot_expert_count = 8

            mock_loaded_state = Mock()
            mock_loaded_state.model_info.is_moe = True

            coordinator_instance.initialize_with_coordination.return_value = (
                mock_vllm_engine,
                mock_allocation,
                mock_loaded_state
            )

            yield mock_cls, coordinator_instance, mock_vllm_engine, mock_allocation, mock_loaded_state

    def test_basic_initialization(self, mock_coordinator):
        """Test basic initialization with minimal parameters."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        # Call the function
        vllm_engine, weight_bridge, allocation = initialize_vllm_with_coordination(
            model_id="mistralai/Mixtral-8x7B-v0.1"
        )

        # Verify coordinator was created correctly
        mock_cls.assert_called_once_with(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            user_vllm_params={},
            storage_path=None
        )

        # Verify coordination was called
        coordinator_instance.initialize_with_coordination.assert_called_once()

        # Verify return values
        assert vllm_engine == mock_engine
        assert isinstance(weight_bridge, VLLMWeightBridge)
        assert allocation == mock_allocation

    def test_initialization_with_vllm_params(self, mock_coordinator):
        """Test initialization with custom vLLM parameters."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        user_params = {
            "max_model_len": 4096,
            "dtype": "bfloat16",
            "quantization": "awq",
            "tensor_parallel_size": 2
        }

        # Call the function
        vllm_engine, weight_bridge, allocation = initialize_vllm_with_coordination(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            user_vllm_params=user_params
        )

        # Verify coordinator was created with correct params
        mock_cls.assert_called_once_with(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            user_vllm_params=user_params,
            storage_path=None
        )

    def test_initialization_with_storage_path(self, mock_coordinator):
        """Test initialization with custom storage path."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        storage_path = "/custom/storage/path"

        # Call the function
        vllm_engine, weight_bridge, allocation = initialize_vllm_with_coordination(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            storage_path=storage_path
        )

        # Verify coordinator was created with storage path
        mock_cls.assert_called_once_with(
            model_id="mistralai/Mixtral-8x7B-v0.1",
            user_vllm_params={},
            storage_path=storage_path
        )

    def test_weight_injection_called(self, mock_coordinator):
        """Test that weight injection is called on the vLLM model."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        # Mock the weight bridge to track injection calls
        with patch.object(VLLMWeightBridge, 'inject_weight_references') as mock_inject:
            # Call the function
            vllm_engine, weight_bridge, allocation = initialize_vllm_with_coordination(
                model_id="mistralai/Mixtral-8x7B-v0.1"
            )

            # Verify inject_weight_references was called
            mock_inject.assert_called_once()
            # Verify it was called with the vLLM model
            args = mock_inject.call_args[0]
            assert args[0] == mock_engine.llm_engine.model_executor.driver_worker.model_runner.model

    def test_weight_injection_with_alternative_vllm_structure(self, mock_coordinator):
        """Test weight injection with alternative vLLM internal structure."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        # Mock alternative vLLM structure (older versions)
        del mock_engine.llm_engine.model_executor  # Remove standard path
        mock_engine.llm_engine.workers = [Mock()]
        mock_engine.llm_engine.workers[0].model = Mock()

        with patch.object(VLLMWeightBridge, 'inject_weight_references') as mock_inject:
            # Call the function
            vllm_engine, weight_bridge, allocation = initialize_vllm_with_coordination(
                model_id="mistralai/Mixtral-8x7B-v0.1"
            )

            # Verify inject_weight_references was called with alternative structure
            mock_inject.assert_called_once()
            args = mock_inject.call_args[0]
            assert args[0] == mock_engine.llm_engine.workers[0].model

    def test_weight_injection_skipped_when_model_not_found(self, mock_coordinator):
        """Test that weight injection is skipped gracefully when model cannot be accessed."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        # Make both model access paths fail
        del mock_engine.llm_engine.model_executor
        mock_engine.llm_engine.workers = []  # Empty workers list

        with patch.object(VLLMWeightBridge, 'inject_weight_references') as mock_inject:
            # Call the function - should not raise
            vllm_engine, weight_bridge, allocation = initialize_vllm_with_coordination(
                model_id="mistralai/Mixtral-8x7B-v0.1"
            )

            # Verify inject_weight_references was NOT called
            mock_inject.assert_not_called()

            # Verify we still got valid return values
            assert vllm_engine == mock_engine
            assert isinstance(weight_bridge, VLLMWeightBridge)
            assert allocation == mock_allocation

    def test_weight_bridge_created_with_loaded_state(self, mock_coordinator):
        """Test that VLLMWeightBridge is created with correct loaded_state."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        # Call the function
        vllm_engine, weight_bridge, allocation = initialize_vllm_with_coordination(
            model_id="mistralai/Mixtral-8x7B-v0.1"
        )

        # Verify weight bridge has correct loaded state
        assert weight_bridge.loaded_state == mock_state

    def test_return_tuple_structure(self, mock_coordinator):
        """Test that return tuple has correct structure and types."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        # Call the function
        result = initialize_vllm_with_coordination(
            model_id="mistralai/Mixtral-8x7B-v0.1"
        )

        # Verify tuple structure
        assert isinstance(result, tuple)
        assert len(result) == 3

        vllm_engine, weight_bridge, allocation = result

        # Verify types
        assert vllm_engine is not None
        assert isinstance(weight_bridge, VLLMWeightBridge)
        assert allocation is not None

    def test_coordinator_initialization_failure_propagates(self, mock_coordinator):
        """Test that coordinator initialization failures are propagated."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        # Make coordinator fail
        coordinator_instance.initialize_with_coordination.side_effect = ValueError(
            "Cannot fulfill memory allocation request"
        )

        # Call should raise
        with pytest.raises(ValueError, match="Cannot fulfill memory allocation request"):
            initialize_vllm_with_coordination(
                model_id="mistralai/Mixtral-8x7B-v0.1"
            )

    def test_dense_model_handling(self, mock_coordinator):
        """Test initialization with dense (non-MoE) model."""
        mock_cls, coordinator_instance, mock_engine, mock_allocation, mock_state = mock_coordinator

        # Set model as dense
        mock_state.model_info.is_moe = False

        # Call the function
        vllm_engine, weight_bridge, allocation = initialize_vllm_with_coordination(
            model_id="meta-llama/Llama-2-7b-hf"
        )

        # Should still work for dense models
        assert vllm_engine == mock_engine
        assert isinstance(weight_bridge, VLLMWeightBridge)
        assert allocation == mock_allocation


class TestVLLMWeightBridgePreservation:
    """Test that existing VLLMWeightBridge functionality is preserved."""

    @pytest.fixture
    def mock_loaded_state(self):
        """Create mock LoadedWeightState."""
        state = Mock()
        state.shared_weights = {
            "embed_tokens.weight": Mock(),
            "model.embed_tokens.weight": Mock()
        }
        state.expert_cache = Mock()
        state.model_info = Mock()
        return state

    def test_weight_bridge_initialization(self, mock_loaded_state):
        """Test that VLLMWeightBridge can still be initialized directly."""
        bridge = VLLMWeightBridge(mock_loaded_state)

        assert bridge.loaded_state == mock_loaded_state
        assert bridge.shared_weights == mock_loaded_state.shared_weights
        assert bridge.expert_cache == mock_loaded_state.expert_cache
        assert bridge.model_info == mock_loaded_state.model_info

    def test_patch_vllm_weight_loading_exists(self, mock_loaded_state):
        """Test that patch_vllm_weight_loading method exists and is callable."""
        bridge = VLLMWeightBridge(mock_loaded_state)
        assert hasattr(bridge, 'patch_vllm_weight_loading')
        assert callable(bridge.patch_vllm_weight_loading)

    def test_inject_weight_references_exists(self, mock_loaded_state):
        """Test that inject_weight_references method exists and is callable."""
        bridge = VLLMWeightBridge(mock_loaded_state)
        assert hasattr(bridge, 'inject_weight_references')
        assert callable(bridge.inject_weight_references)

    def test_setup_expert_swap_hook_exists(self, mock_loaded_state):
        """Test that setup_expert_swap_hook method exists and is callable."""
        bridge = VLLMWeightBridge(mock_loaded_state)
        assert hasattr(bridge, 'setup_expert_swap_hook')
        assert callable(bridge.setup_expert_swap_hook)

    def test_get_shared_weight_reference_exists(self, mock_loaded_state):
        """Test that _get_shared_weight_reference method exists and is callable."""
        bridge = VLLMWeightBridge(mock_loaded_state)
        assert hasattr(bridge, '_get_shared_weight_reference')
        assert callable(bridge._get_shared_weight_reference)

    def test_get_expert_weight_reference_exists(self, mock_loaded_state):
        """Test that _get_expert_weight_reference method exists and is callable."""
        bridge = VLLMWeightBridge(mock_loaded_state)
        assert hasattr(bridge, '_get_expert_weight_reference')
        assert callable(bridge._get_expert_weight_reference)

    def test_parse_expert_name_exists(self, mock_loaded_state):
        """Test that _parse_expert_name method exists and is callable."""
        bridge = VLLMWeightBridge(mock_loaded_state)
        assert hasattr(bridge, '_parse_expert_name')
        assert callable(bridge._parse_expert_name)


class TestModuleDocstring:
    """Test that module docstring is updated correctly."""

    def test_module_has_docstring(self):
        """Test that vllm_bridge module has a docstring."""
        import sparse_llm.integrations.vllm_bridge as bridge_module
        assert bridge_module.__doc__ is not None
        assert len(bridge_module.__doc__) > 0

    def test_docstring_mentions_two_phase_approach(self):
        """Test that docstring mentions the two-phase approach."""
        import sparse_llm.integrations.vllm_bridge as bridge_module
        docstring = bridge_module.__doc__.lower()

        assert "phase 1" in docstring or "phase 2" in docstring
        assert "memory" in docstring
        assert "coordinator" in docstring or "coordination" in docstring

    def test_docstring_mentions_both_approaches(self):
        """Test that docstring describes both usage approaches."""
        import sparse_llm.integrations.vllm_bridge as bridge_module
        docstring = bridge_module.__doc__.lower()

        # Should mention coordinated initialization
        assert "initialize_vllm_with_coordination" in docstring

        # Should mention manual weight bridge
        assert "vllmweightbridge" in docstring

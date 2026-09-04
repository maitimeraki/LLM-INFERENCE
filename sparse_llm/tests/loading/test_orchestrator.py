"""Tests for FourPhaseOrchestrator."""

import pytest
from unittest.mock import Mock, patch
import torch


def test_four_phase_orchestrator_orchestration():
    """Test complete orchestration of all 4 phases."""
    # Mock progress callback
    progress_callback = Mock()

    # Mock all phases (won't actually load model)
    with patch('sparse_llm.loading.orchestrator.ResourceProfiler') as MockProfiler, \
         patch('sparse_llm.loading.orchestrator.ModelIntrospector') as MockIntrospector, \
         patch('sparse_llm.loading.orchestrator.PlacementStrategyCalculator') as MockCalculator, \
         patch('sparse_llm.loading.orchestrator.WeightLoader') as MockLoader:

        from sparse_llm.loading.orchestrator import FourPhaseOrchestrator
        orchestrator = FourPhaseOrchestrator()

        # Setup mocks
        from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo
        from sparse_llm.loading.model_introspector import ModelInfo
        from sparse_llm.loading.placement_plan import PlacementPlan
        from sparse_llm.loading.loaded_weight_state import LoadedWeightState
        from sparse_llm.loading.expert_cache import ExpertCache

        mock_budget = ResourceBudget(
            gpus=[GPUInfo(0, 24*1024**3, 20*1024**3, (8,0), "A100")],
            cpu=CPUInfo(64*1024**3, 48*1024**3, 38*1024**3),
            storage=StorageInfo("/cache", 500*1024**3, True, 1200)
        )

        mock_model_info = ModelInfo(
            model_id="test", is_moe=True, num_layers=2, num_experts=4,
            num_experts_per_tok=2, shared_weight_bytes=1024, expert_weight_bytes=512, total_bytes=2048
        )

        mock_plan = PlacementPlan(
            model_info=mock_model_info, shared_device="cuda:0", hot_expert_slots=4,
            warm_expert_slots=2, cold_expert_count=2, gpu_utilization_pct=70.0,
            cpu_utilization_pct=30.0, estimated_load_time_sec=45.0
        )

        mock_state = LoadedWeightState(
            shared_weights={"test": torch.randn(10, 10)},
            expert_cache=ExpertCache(4, 2, 512),
            model_info=mock_model_info,
            placement_plan=mock_plan
        )

        MockProfiler.return_value.profile.return_value = mock_budget
        MockIntrospector.return_value.introspect.return_value = mock_model_info
        MockCalculator.return_value.calculate.return_value = mock_plan
        MockLoader.return_value.load.return_value = mock_state

        # Run initialization
        state = orchestrator.initialize("test-model", progress_callback=progress_callback)

        # Verify all phases called
        MockProfiler.return_value.profile.assert_called_once()
        MockIntrospector.return_value.introspect.assert_called_once_with("test-model")
        MockCalculator.return_value.calculate.assert_called_once()
        MockLoader.return_value.load.assert_called_once()

        # Verify progress callback received phase messages
        assert progress_callback.call_count >= 4  # At least one message per phase

        # Verify returned state
        assert state.model_info == mock_model_info
        assert state.placement_plan == mock_plan


def test_orchestrator_passes_storage_path():
    """Test that custom storage_path is passed through phases."""
    with patch('sparse_llm.loading.orchestrator.ResourceProfiler') as MockProfiler, \
         patch('sparse_llm.loading.orchestrator.ModelIntrospector') as MockIntrospector, \
         patch('sparse_llm.loading.orchestrator.PlacementStrategyCalculator') as MockCalculator, \
         patch('sparse_llm.loading.orchestrator.WeightLoader') as MockLoader:

        from sparse_llm.loading.orchestrator import FourPhaseOrchestrator
        orchestrator = FourPhaseOrchestrator()

        from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo
        from sparse_llm.loading.model_introspector import ModelInfo
        from sparse_llm.loading.placement_plan import PlacementPlan
        from sparse_llm.loading.loaded_weight_state import LoadedWeightState
        from sparse_llm.loading.expert_cache import ExpertCache

        mock_budget = ResourceBudget(
            gpus=[GPUInfo(0, 24*1024**3, 20*1024**3, (8,0), "A100")],
            cpu=CPUInfo(64*1024**3, 48*1024**3, 38*1024**3),
            storage=StorageInfo("/custom/path", 500*1024**3, True, 1200)
        )

        mock_model_info = ModelInfo(
            model_id="test", is_moe=True, num_layers=2, num_experts=4,
            num_experts_per_tok=2, shared_weight_bytes=1024, expert_weight_bytes=512, total_bytes=2048
        )

        mock_plan = PlacementPlan(
            model_info=mock_model_info, shared_device="cuda:0", hot_expert_slots=4,
            warm_expert_slots=2, cold_expert_count=2, gpu_utilization_pct=70.0,
            cpu_utilization_pct=30.0, estimated_load_time_sec=45.0
        )

        mock_state = LoadedWeightState(
            shared_weights={"test": torch.randn(10, 10)},
            expert_cache=ExpertCache(4, 2, 512),
            model_info=mock_model_info,
            placement_plan=mock_plan
        )

        MockProfiler.return_value.profile.return_value = mock_budget
        MockIntrospector.return_value.introspect.return_value = mock_model_info
        MockCalculator.return_value.calculate.return_value = mock_plan
        MockLoader.return_value.load.return_value = mock_state

        # Run with custom storage path
        state = orchestrator.initialize("test-model", storage_path="/custom/path")

        # Verify storage_path was passed to profiler
        MockProfiler.return_value.profile.assert_called_once_with(storage_path="/custom/path")

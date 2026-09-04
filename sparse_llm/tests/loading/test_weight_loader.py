"""Tests for WeightLoader."""

def test_weight_loader_mock_load():
    from sparse_llm.loading.weight_loader import WeightLoader
    from sparse_llm.loading.placement_plan import PlacementPlan
    from sparse_llm.loading.model_introspector import ModelInfo
    from unittest.mock import Mock, patch, MagicMock
    from pathlib import Path

    model_info = ModelInfo(
        model_id="test-model", is_moe=True, num_layers=2, num_experts=4,
        num_experts_per_tok=2, shared_weight_bytes=1*1024**3,
        expert_weight_bytes=100*1024**2, total_bytes=2*1024**3
    )

    plan = PlacementPlan(
        model_info=model_info, shared_device="cpu", hot_expert_slots=2,
        warm_expert_slots=1, cold_expert_count=5, gpu_utilization_pct=25.0,
        cpu_utilization_pct=30.0, estimated_load_time_sec=45.0
    )

    loader = WeightLoader()

    # Mock progress callback
    progress_callback = Mock()

    # Mock Path.exists and snapshot_download
    with patch('sparse_llm.loading.weight_loader.Path') as mock_path_cls, \
         patch('sparse_llm.loading.weight_loader.snapshot_download') as mock_snapshot, \
         patch('sparse_llm.loading.weight_loader.safe_open') as mock_safe_open:

        # Mock Path behavior
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.glob.return_value = [Path("model.safetensors")]
        mock_path_cls.return_value = mock_path

        # Mock safetensors file context manager
        import torch
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.keys.return_value = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.0.block_sparse_moe.experts.1.w1.weight",
            "model.layers.0.block_sparse_moe.experts.1.w2.weight",
        ]
        mock_file.get_tensor.return_value = torch.randn(100, 100)
        mock_safe_open.return_value = mock_file

        # Load
        state = loader.load("test-model", plan, progress_callback=progress_callback)

        # Verify progress callback was called
        assert progress_callback.called

        # Verify state structure
        assert state.model_info == model_info
        assert state.placement_plan == plan
        assert state.shared_weights is not None
        assert state.expert_cache is not None


def test_weight_loader_identifies_shared_vs_expert():
    from sparse_llm.loading.weight_loader import WeightLoader

    loader = WeightLoader()

    # Test shared weight pattern
    assert loader._is_shared_weight("model.embed_tokens.weight") == True
    assert loader._is_shared_weight("model.layers.0.self_attn.q_proj.weight") == True
    assert loader._is_shared_weight("model.layers.5.input_layernorm.weight") == True

    # Test expert weight pattern
    assert loader._is_expert_weight("model.layers.0.block_sparse_moe.experts.3.w1.weight") == True
    assert loader._is_expert_weight("model.layers.2.mlp.experts.7.gate_proj.weight") == True

    # Not expert if no "experts" in name
    assert loader._is_expert_weight("model.layers.0.mlp.w1.weight") == False

"""Tests for Mixtral adapter and paging capabilities."""

from types import SimpleNamespace

import pytest

from sparse_llm.models.mixtral_adapter import (
    MixtralAdapter,
    MixtralPagingMetadata,
    is_mixtral_config,
)
from sparse_llm.models.paging import PagingCapabilities
from sparse_llm.models.registry import get_default_registry


class TestMixtralConfigDetection:
    """Test Mixtral config detection predicate."""

    def test_mixtral_config_is_detected(self):
        """Predicate should detect Mixtral model type."""
        config = SimpleNamespace(model_type="mixtral")
        assert is_mixtral_config(config) is True

    def test_non_mixtral_config_is_rejected(self):
        """Predicate should reject non-Mixtral models."""
        config = SimpleNamespace(model_type="llama")
        assert is_mixtral_config(config) is False

    def test_missing_model_type_is_rejected(self):
        """Predicate should handle missing model_type gracefully."""
        config = SimpleNamespace()
        assert is_mixtral_config(config) is False

    def test_none_config_is_rejected(self):
        """Predicate should handle None config."""
        assert is_mixtral_config(None) is False


class TestMixtralPagingMetadata:
    """Test Mixtral paging metadata extraction."""

    def test_mixtral_paging_metadata_initialization(self):
        """Metadata should initialize with correct topology values."""
        metadata = MixtralPagingMetadata(
            num_hidden_layers=32,
            num_local_experts=8,
            num_experts_per_tok=2,
        )
        assert metadata.num_hidden_layers == 32
        assert metadata.num_local_experts == 8
        assert metadata.num_experts_per_tok == 2

    def test_mixtral_router_tensor_names_generation(self):
        """Router tensor names should follow Mixtral naming convention."""
        metadata = MixtralPagingMetadata(
            num_hidden_layers=2,
            num_local_experts=8,
            num_experts_per_tok=2,
        )
        names = metadata.router_tensor_names
        assert len(names) == 2
        assert names[0] == "model.layers.0.block_sparse_moe.gate"
        assert names[1] == "model.layers.1.block_sparse_moe.gate"

    def test_mixtral_expert_tensor_names_generation(self):
        """Expert tensor names should cover all layers and experts."""
        metadata = MixtralPagingMetadata(
            num_hidden_layers=2,
            num_local_experts=2,
            num_experts_per_tok=2,
        )
        names = metadata.expert_tensor_names
        # 2 layers * 2 experts * 3 weights (w1, w2, w3) = 12
        assert len(names) == 12
        assert names[0] == "model.layers.0.block_sparse_moe.experts.0.w1"
        assert names[1] == "model.layers.0.block_sparse_moe.experts.0.w2"
        assert names[2] == "model.layers.0.block_sparse_moe.experts.0.w3"
        assert names[3] == "model.layers.0.block_sparse_moe.experts.1.w1"
        # Check that we have correct structure for layer 1
        assert "model.layers.1.block_sparse_moe.experts" in names[6]


class TestMixtralAdapter:
    """Test MixtralAdapter functionality."""

    def test_mixtral_adapter_extracts_topology(self):
        """Adapter should extract layer and expert topology from config."""
        config = SimpleNamespace(
            model_type="mixtral",
            num_hidden_layers=32,
            num_local_experts=8,
            num_experts_per_tok=2,
        )
        adapter = MixtralAdapter(
            "mistralai/Mixtral-8x7B-v0.1",
            config=config,
        )
        assert adapter._paging_metadata is not None
        assert adapter._paging_metadata.num_hidden_layers == 32
        assert adapter._paging_metadata.num_local_experts == 8
        assert adapter._paging_metadata.num_experts_per_tok == 2

    def test_mixtral_adapter_handles_missing_config(self):
        """Adapter should handle None config gracefully."""
        adapter = MixtralAdapter(
            "mistralai/Mixtral-8x7B-v0.1",
            config=None,
        )
        assert adapter._paging_metadata is None

    def test_mixtral_adapter_builds_paging_capabilities(self):
        """Adapter should provide paging capabilities when metadata exists."""
        config = SimpleNamespace(
            model_type="mixtral",
            num_hidden_layers=32,
            num_local_experts=8,
            num_experts_per_tok=2,
        )
        adapter = MixtralAdapter(
            "mistralai/Mixtral-8x7B-v0.1",
            config=config,
        )
        capabilities = adapter.paging_capabilities
        assert capabilities is not None
        assert isinstance(capabilities, PagingCapabilities)
        assert capabilities.num_layers == 32
        assert capabilities.num_experts == 8
        assert capabilities.top_k == 2
        assert capabilities.validation_passed is True
        assert capabilities.supports_prefill is True
        assert capabilities.supports_decode is True

    def test_mixtral_adapter_no_paging_capabilities_without_metadata(self):
        """Adapter without metadata should return None for paging capabilities."""
        adapter = MixtralAdapter(
            "mistralai/Mixtral-8x7B-v0.1",
            config=None,
        )
        assert adapter.paging_capabilities is None

    def test_mixtral_adapter_has_correct_architecture_classification(self):
        """Adapter should report MoE architecture classification."""
        config = SimpleNamespace(
            model_type="mixtral",
            num_hidden_layers=32,
            num_local_experts=8,
            num_experts_per_tok=2,
        )
        adapter = MixtralAdapter(
            "mistralai/Mixtral-8x7B-v0.1",
            config=config,
        )
        capabilities = adapter.capabilities
        assert capabilities.is_moe is True
        assert capabilities.expert_paging is True
        assert capabilities.architecture_classification == "moe"


class TestMixtralRegistration:
    """Test Mixtral registration in the default registry."""

    def test_registry_selects_mixtral_adapter_by_priority(self):
        """Registry should select MixtralAdapter over generic for Mixtral config."""
        registry = get_default_registry()
        config = SimpleNamespace(model_type="mixtral")
        adapter = registry.create(
            "mistralai/Mixtral-8x7B-v0.1",
            config=config,
        )
        assert isinstance(adapter, MixtralAdapter)

    def test_registry_falls_back_to_generic_without_config(self):
        """Registry should use generic adapter when config is unavailable."""
        from sparse_llm.models.adapters import TransformersCausalLMAdapter

        registry = get_default_registry()
        adapter = registry.create(
            "mistralai/Mixtral-8x7B-v0.1",
        )
        # Should fall back to generic when no config provided
        assert isinstance(adapter, TransformersCausalLMAdapter)

    def test_registry_includes_mixtral_in_names(self):
        """Registry should include 'mixtral' in its registered adapter names."""
        registry = get_default_registry()
        names = registry.names()
        assert "mixtral" in names


class TestPagingCapabilitiesFactory:
    """Test PagingCapabilities.from_mixtral() factory method."""

    def test_from_mixtral_creates_valid_capabilities(self):
        """Factory should create valid PagingCapabilities from metadata."""
        metadata = MixtralPagingMetadata(
            num_hidden_layers=32,
            num_local_experts=8,
            num_experts_per_tok=2,
        )
        capabilities = PagingCapabilities.from_mixtral(metadata)
        assert capabilities.architecture_id == "mixtral-8x7b"
        assert capabilities.adapter_version == "1.0"
        assert capabilities.num_layers == 32
        assert capabilities.num_experts == 8
        assert capabilities.top_k == 2
        assert capabilities.validation_passed is True
        assert capabilities.supports_prefill is True
        assert capabilities.supports_decode is True

    def test_from_mixtral_includes_expert_tensor_names(self):
        """Factory should include all expert tensor names in capabilities."""
        metadata = MixtralPagingMetadata(
            num_hidden_layers=2,
            num_local_experts=2,
            num_experts_per_tok=2,
        )
        capabilities = PagingCapabilities.from_mixtral(metadata)
        tensor_names = capabilities.expert_tensor_names
        assert len(tensor_names) == 12  # 2 layers * 2 experts * 3 weights
        assert "model.layers.0.block_sparse_moe.experts.0.w1" in tensor_names
        assert "model.layers.1.block_sparse_moe.experts.1.w3" in tensor_names

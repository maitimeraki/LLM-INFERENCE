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
from sparse_llm.models.shared_weight_loader import SharedWeightPlacer


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


class TestSharedWeightPlacer:
    """Test weight classification for Mixtral models."""

    def test_placer_classifies_shared_weights(self):
        """Placer should classify embedding and attention weights as shared."""
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_local_experts=4,
        )
        placer = SharedWeightPlacer(config)
        assert placer.classify_tensor("model.embed_tokens.weight") == "shared"
        assert placer.classify_tensor("model.layers.0.self_attn.q_proj.weight") == "shared"
        assert placer.classify_tensor("model.layers.1.input_layernorm.weight") == "shared"

    def test_placer_classifies_expert_weights(self):
        """Placer should classify expert weights as expert."""
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_local_experts=4,
        )
        placer = SharedWeightPlacer(config)
        assert placer.classify_tensor("model.layers.0.block_sparse_moe.experts.0.w1.weight") == "expert"
        assert placer.classify_tensor("model.layers.1.block_sparse_moe.experts.3.w2.weight") == "expert"

    def test_placer_classifies_unknown_weights_as_other(self):
        """Placer should classify unknown weights as other."""
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_local_experts=4,
        )
        placer = SharedWeightPlacer(config)
        assert placer.classify_tensor("unknown.weight") == "other"

    def test_placer_generates_shared_weights_set(self):
        """Placer should generate complete set of shared weight names."""
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_local_experts=4,
        )
        placer = SharedWeightPlacer(config)
        shared = placer.shared_weight_names()
        # Check key shared weights
        assert "model.embed_tokens.weight" in shared
        assert "model.norm.weight" in shared
        assert "lm_head.weight" in shared
        assert "model.layers.0.self_attn.q_proj.weight" in shared
        assert "model.layers.0.block_sparse_moe.gate" in shared  # Router
        assert "model.layers.1.input_layernorm.weight" in shared
        # Verify no expert weights in shared set
        assert "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in shared

    def test_placer_generates_expert_weights_set(self):
        """Placer should generate complete set of expert weight names."""
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_local_experts=4,
        )
        placer = SharedWeightPlacer(config)
        experts = placer.expert_weight_names()
        # 2 layers * 4 experts * 3 weights = 24
        assert len(experts) == 24
        assert "model.layers.0.block_sparse_moe.experts.0.w1.weight" in experts
        assert "model.layers.1.block_sparse_moe.experts.3.w3.weight" in experts

    def test_placer_separates_shared_and_expert_weights(self):
        """Placer should ensure no overlap between shared and expert weights."""
        config = SimpleNamespace(
            num_hidden_layers=2,
            num_local_experts=4,
        )
        placer = SharedWeightPlacer(config)
        shared = placer.shared_weight_names()
        experts = placer.expert_weight_names()
        overlap = shared & experts
        assert len(overlap) == 0, f"Shared and expert weights should not overlap, but found: {overlap}"



class TestMixtralIntegration:
    """End-to-end integration tests with real tokenizer and model (optional GPU)."""

    @pytest.mark.skip(reason="Requires optional GPU and model download")
    def test_mixtral_adapter_generates_text_with_real_tokenizer_and_model(self):
        """Integration test: real text generation with Mixtral and paging metrics.

        This test requires:
        - Transformers library installed
        - Mixtral-8x7B model available or downloadable
        - GPU with sufficient VRAM (optional, falls back to CPU)

        Skipped by default to avoid long test times and model downloads.
        Run with: pytest --run-integration sparse_llm/tests/test_mixtral_adapter.py::TestMixtralIntegration
        """
        try:
            from sparse_llm import InferenceEngine
        except ImportError:
            pytest.skip("sparse_llm not available")

        try:
            # Use a small test model identifier or skip if unavailable
            adapter = MixtralAdapter("mistralai/Mixtral-8x7B-v0.1")
            adapter.load()
        except Exception as e:
            pytest.skip(f"Model unavailable or loading failed: {e}")

        # Real generation with paging infrastructure in place
        prompt = "The future of AI is"
        result = adapter.generate(prompt, max_new_tokens=16, temperature=0.0)

        # Validate real text was generated
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.text != prompt  # Should have generated something

        # Validate metrics are populated
        assert result.metrics.prompt_tokens > 0
        assert result.metrics.generated_tokens > 0
        assert result.metrics.prefill_latency_ms >= 0.0
        assert result.metrics.decode_latency_ms >= 0.0

        # Paging metrics present (even if zeros for parent-called generation)
        assert isinstance(result.metrics.cache_hits, int)
        assert isinstance(result.metrics.cache_misses, int)
        assert isinstance(result.metrics.expert_load_time_ms, float)

        # Validate token IDs match decoded text
        assert len(result.token_ids) > 0
        assert result.token_ids[0] >= 0




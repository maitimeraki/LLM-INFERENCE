"""Tests for paged generation runner and per-layer metrics collection."""

import pytest
import torch

from sparse_llm.cache import ExpertCache
from sparse_llm.inference.paged_generation import PagedGenerationRunner, LayerPageStats


class TestLayerPageStats:
    """Test per-layer paging metrics dataclass."""

    def test_layer_page_stats_initialization(self):
        """Stats should initialize with correct layer and expert count."""
        stats = LayerPageStats(layer_idx=2, num_experts_requested=4)
        assert stats.layer_idx == 2
        assert stats.num_experts_requested == 4
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.expert_load_time_ms == 0.0

    def test_layer_page_stats_to_dict(self):
        """Stats should serialize to dictionary correctly."""
        stats = LayerPageStats(
            layer_idx=1,
            num_experts_requested=3,
            cache_hits=2,
            cache_misses=1,
            expert_load_time_ms=5.5,
        )
        d = stats.to_dict()
        assert d["layer_idx"] == 1
        assert d["num_experts_requested"] == 3
        assert d["cache_hits"] == 2
        assert d["cache_misses"] == 1
        assert d["expert_load_time_ms"] == 5.5


class TestPagedGenerationRunner:
    """Test PagedGenerationRunner initialization and validation."""

    def test_runner_initialization(self):
        """Runner should initialize with layer/expert topology and cache."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=32, num_experts=8, cache=cache)
        assert runner.num_layers == 32
        assert runner.num_experts == 8
        assert runner.cache is cache
        assert runner.per_layer_stats == {}

    def test_runner_rejects_invalid_num_layers(self):
        """Runner should reject invalid num_layers."""
        cache = ExpertCache(max_experts=8)
        with pytest.raises(ValueError, match="num_layers"):
            PagedGenerationRunner(num_layers=0, num_experts=8, cache=cache)
        with pytest.raises(ValueError, match="num_layers"):
            PagedGenerationRunner(num_layers=-1, num_experts=8, cache=cache)

    def test_runner_rejects_invalid_num_experts(self):
        """Runner should reject invalid num_experts."""
        cache = ExpertCache(max_experts=8)
        with pytest.raises(ValueError, match="num_experts"):
            PagedGenerationRunner(num_layers=32, num_experts=0, cache=cache)
        with pytest.raises(ValueError, match="num_experts"):
            PagedGenerationRunner(num_layers=32, num_experts=-1, cache=cache)

    def test_runner_rejects_none_cache(self):
        """Runner should reject None cache."""
        with pytest.raises(ValueError, match="cache"):
            PagedGenerationRunner(num_layers=32, num_experts=8, cache=None)


class TestPagedLayerForward:
    """Test paged layer forward pass with cache checking."""

    def test_paged_layer_forward_tracks_cache_hits(self):
        """Forward pass should record cache hits for cached experts."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=2, num_experts=8, cache=cache)

        # Pre-populate cache with some experts
        cache.put((0, 0), torch.randn(4096, 14336))
        cache.put((0, 1), torch.randn(4096, 14336))

        # Forward pass requesting cached experts
        hidden = torch.randn(8, 2048, 4096)
        output = runner.forward_paged_layer(
            layer_idx=0, expert_ids=[0, 1], hidden_state=hidden
        )

        # Check output shape matches input
        assert output.shape == hidden.shape

        # Check stats recorded 2 hits, 0 misses
        stats = runner.per_layer_stats[0]
        assert stats.cache_hits == 2
        assert stats.cache_misses == 0

    def test_paged_layer_forward_tracks_cache_misses(self):
        """Forward pass should record cache misses for uncached experts."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=2, num_experts=8, cache=cache)

        hidden = torch.randn(8, 2048, 4096)
        output = runner.forward_paged_layer(
            layer_idx=0, expert_ids=[0, 1, 2], hidden_state=hidden
        )

        # Check stats recorded 0 hits, 3 misses
        stats = runner.per_layer_stats[0]
        assert stats.cache_hits == 0
        assert stats.cache_misses == 3

    def test_paged_layer_forward_mixed_hits_and_misses(self):
        """Forward pass should correctly record mix of hits and misses."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=2, num_experts=8, cache=cache)

        # Cache experts 0 and 2
        cache.put((0, 0), torch.randn(4096, 14336))
        cache.put((0, 2), torch.randn(4096, 14336))

        hidden = torch.randn(8, 2048, 4096)
        output = runner.forward_paged_layer(
            layer_idx=0, expert_ids=[0, 1, 2, 3], hidden_state=hidden
        )

        # Should have 2 hits (0, 2) and 2 misses (1, 3)
        stats = runner.per_layer_stats[0]
        assert stats.cache_hits == 2
        assert stats.cache_misses == 2

    def test_paged_layer_forward_rejects_invalid_layer_idx(self):
        """Forward pass should reject invalid layer index."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=2, num_experts=8, cache=cache)
        hidden = torch.randn(8, 2048, 4096)

        with pytest.raises(ValueError, match="layer_idx"):
            runner.forward_paged_layer(layer_idx=-1, expert_ids=[0], hidden_state=hidden)

        with pytest.raises(ValueError, match="layer_idx"):
            runner.forward_paged_layer(layer_idx=2, expert_ids=[0], hidden_state=hidden)

    def test_paged_layer_forward_rejects_invalid_expert_ids(self):
        """Forward pass should reject invalid expert IDs."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=2, num_experts=8, cache=cache)
        hidden = torch.randn(8, 2048, 4096)

        with pytest.raises(ValueError, match="expert_ids"):
            runner.forward_paged_layer(layer_idx=0, expert_ids="invalid", hidden_state=hidden)

        with pytest.raises(ValueError, match="expert_ids"):
            runner.forward_paged_layer(layer_idx=0, expert_ids=[-1], hidden_state=hidden)

        with pytest.raises(ValueError, match="expert_ids"):
            runner.forward_paged_layer(layer_idx=0, expert_ids=[8], hidden_state=hidden)

    def test_paged_layer_forward_rejects_non_tensor_hidden_state(self):
        """Forward pass should reject non-tensor hidden state."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=2, num_experts=8, cache=cache)

        with pytest.raises(ValueError, match="hidden_state"):
            runner.forward_paged_layer(
                layer_idx=0, expert_ids=[0], hidden_state="not_a_tensor"
            )


class TestPagedGenerationRunnerStats:
    """Test aggregated metrics collection."""

    def test_stats_dict_empty_when_no_layers_executed(self):
        """Stats should be empty aggregates before any forward passes."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=2, num_experts=8, cache=cache)

        stats = runner.stats_dict()
        assert stats["per_layer"] == {}
        assert stats["aggregated"]["total_experts_requested"] == 0
        assert stats["aggregated"]["total_cache_hits"] == 0
        assert stats["aggregated"]["total_cache_misses"] == 0
        assert stats["aggregated"]["total_cache_accesses"] == 0
        assert stats["aggregated"]["hit_rate"] == 0.0

    def test_stats_dict_single_layer_aggregation(self):
        """Stats should aggregate metrics from a single layer."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=2, num_experts=8, cache=cache)

        # Populate cache with 1 expert
        cache.put((0, 0), torch.randn(4096, 14336))

        # Forward pass requesting 3 experts (1 hit, 2 misses)
        hidden = torch.randn(8, 2048, 4096)
        runner.forward_paged_layer(layer_idx=0, expert_ids=[0, 1, 2], hidden_state=hidden)

        stats = runner.stats_dict()
        assert "layer_0" in stats["per_layer"]
        assert stats["per_layer"]["layer_0"]["num_experts_requested"] == 3
        assert stats["per_layer"]["layer_0"]["cache_hits"] == 1
        assert stats["per_layer"]["layer_0"]["cache_misses"] == 2

        assert stats["aggregated"]["total_experts_requested"] == 3
        assert stats["aggregated"]["total_cache_hits"] == 1
        assert stats["aggregated"]["total_cache_misses"] == 2
        assert stats["aggregated"]["total_cache_accesses"] == 3
        assert stats["aggregated"]["hit_rate"] == pytest.approx(1 / 3)

    def test_stats_dict_multiple_layer_aggregation(self):
        """Stats should aggregate metrics across multiple layers."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=3, num_experts=8, cache=cache)

        # Layer 0: cache expert 0, request [0, 1] → 1 hit, 1 miss
        cache.put((0, 0), torch.randn(4096, 14336))
        hidden = torch.randn(8, 2048, 4096)
        runner.forward_paged_layer(layer_idx=0, expert_ids=[0, 1], hidden_state=hidden)

        # Layer 1: cache experts 0, 1, request [0, 1, 2] → 2 hits, 1 miss
        cache.put((1, 0), torch.randn(4096, 14336))
        cache.put((1, 1), torch.randn(4096, 14336))
        runner.forward_paged_layer(layer_idx=1, expert_ids=[0, 1, 2], hidden_state=hidden)

        # Layer 2: request [3, 4] → 0 hits, 2 misses
        runner.forward_paged_layer(layer_idx=2, expert_ids=[3, 4], hidden_state=hidden)

        stats = runner.stats_dict()

        # Check per-layer stats
        assert stats["per_layer"]["layer_0"]["cache_hits"] == 1
        assert stats["per_layer"]["layer_0"]["cache_misses"] == 1
        assert stats["per_layer"]["layer_1"]["cache_hits"] == 2
        assert stats["per_layer"]["layer_1"]["cache_misses"] == 1
        assert stats["per_layer"]["layer_2"]["cache_hits"] == 0
        assert stats["per_layer"]["layer_2"]["cache_misses"] == 2

        # Check aggregated stats
        assert stats["aggregated"]["total_cache_hits"] == 3
        assert stats["aggregated"]["total_cache_misses"] == 4
        assert stats["aggregated"]["total_cache_accesses"] == 7
        assert stats["aggregated"]["total_experts_requested"] == 7
        assert stats["aggregated"]["hit_rate"] == pytest.approx(3 / 7)

    def test_stats_dict_hit_rate_calculation(self):
        """Stats should calculate hit rate correctly."""
        cache = ExpertCache(max_experts=8)
        runner = PagedGenerationRunner(num_layers=1, num_experts=8, cache=cache)

        # Cache all 4 experts
        for i in range(4):
            cache.put((0, i), torch.randn(4096, 14336))

        hidden = torch.randn(8, 2048, 4096)
        runner.forward_paged_layer(layer_idx=0, expert_ids=[0, 1, 2, 3], hidden_state=hidden)

        stats = runner.stats_dict()
        assert stats["aggregated"]["hit_rate"] == pytest.approx(1.0)


class TestPagedGenerationIntegration:
    """Integration tests with realistic cache scenarios."""

    def test_paged_generation_with_mock_cache(self):
        """Runner should work with mock cache interface."""
        cache = ExpertCache(max_experts=16)
        runner = PagedGenerationRunner(num_layers=4, num_experts=8, cache=cache)

        # Simulate multiple forward passes
        for layer in range(4):
            # Cache half the experts for each layer
            for expert in range(4):
                cache.put((layer, expert), torch.randn(4096, 14336))

            # Request all experts
            hidden = torch.randn(4, 2048, 4096)
            runner.forward_paged_layer(
                layer_idx=layer, expert_ids=list(range(8)), hidden_state=hidden
            )

        stats = runner.stats_dict()

        # Should have 4 layers * 4 cached experts = 16 hits
        # And 4 layers * 4 uncached experts = 16 misses
        assert stats["aggregated"]["total_cache_hits"] == 16
        assert stats["aggregated"]["total_cache_misses"] == 16
        assert stats["aggregated"]["hit_rate"] == pytest.approx(0.5)

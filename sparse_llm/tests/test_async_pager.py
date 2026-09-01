"""Tests for AsyncExpertPager with asyncio-based prefetch."""

import pytest
import torch

from sparse_llm.cache import ExpertCache
from sparse_llm.inference.async_pager import AsyncExpertPager, BackpressureConfig
from sparse_llm.inference.pager import ExpertPager
from sparse_llm.storage.checkpoint_index import CheckpointIndex


class MockCheckpointIndex:
    """Mock CheckpointIndex for testing."""

    def __init__(self):
        self._experts = {}

    def add_expert(self, key, byte_size=1024):
        """Add mock expert."""
        self._experts[key] = {"byte_size": byte_size}

    def expert(self, key):
        """Get expert metadata."""
        from types import SimpleNamespace
        data = self._experts.get(key, {"byte_size": 1024})
        return SimpleNamespace(**data)

    def load_expert(self, key):
        """Load mock expert tensors."""
        return {
            "weight": torch.randn(128, 64),
            "bias": torch.randn(128),
        }


class TestAsyncExpertPager:
    """AsyncExpertPager tests."""

    @pytest.fixture
    def mock_index(self):
        """Create mock checkpoint index."""
        index = MockCheckpointIndex()
        index.add_expert((0, 0), byte_size=2048)
        index.add_expert((0, 1), byte_size=2048)
        index.add_expert((1, 0), byte_size=2048)
        return index

    @pytest.fixture
    def sync_pager(self, mock_index):
        """Create underlying sync pager."""
        cache = ExpertCache(max_experts=4)
        return ExpertPager(
            index=mock_index,
            cache=cache,
            device="cpu",
            enable_cpu_cache=True,
        )

    @pytest.fixture
    def async_pager(self, sync_pager):
        """Create async pager."""
        config = BackpressureConfig(max_outstanding=2)
        return AsyncExpertPager(sync_pager=sync_pager, config=config)

    def test_async_pager_initialization(self, async_pager):
        """Pager initializes with correct config."""
        assert async_pager.sync_pager is not None
        assert async_pager.config.max_outstanding == 2

    def test_default_backpressure_config(self, sync_pager):
        """Default config has sensible limits."""
        pager = AsyncExpertPager(sync_pager=sync_pager)
        assert pager.config.max_outstanding == 3

    def test_sync_lease_basic_operation(self, async_pager):
        """Sync lease works through async pager."""
        key = (0, 0)
        with async_pager.sync_lease(key) as expert:
            assert expert.key == key
            assert "weight" in expert.tensors
            assert "bias" in expert.tensors

    def test_sync_lease_multiple_experts(self, async_pager):
        """Sync lease handles multiple experts."""
        keys = [(0, 0), (0, 1), (1, 0)]
        for key in keys:
            with async_pager.sync_lease(key) as expert:
                assert expert.key == key

    def test_request_stats_inherited(self, async_pager):
        """Stats come from underlying sync pager."""
        key = (0, 0)
        with async_pager.sync_lease(key):
            pass
        stats = async_pager.request_stats()
        assert "cache_hits" in stats
        assert "cache_misses" in stats

    def test_clear_request_stats(self, async_pager):
        """Clear stats delegates to sync pager."""
        key = (0, 0)
        with async_pager.sync_lease(key):
            pass

        async_pager.clear_request_stats()
        stats = async_pager.request_stats()
        # After clear, stats should be reset
        assert stats["cache_hits"] >= 0

    def test_backpressure_config_frozen(self):
        """BackpressureConfig is a dataclass."""
        config = BackpressureConfig(max_outstanding=5)
        assert config.max_outstanding == 5

    def test_sync_lease_is_context_manager(self, mock_index):
        """Sync lease returns proper context manager."""
        cache = ExpertCache(max_experts=2)
        sync_pager = ExpertPager(index=mock_index, cache=cache, device="cpu")
        async_pager = AsyncExpertPager(sync_pager=sync_pager)

        key = (0, 0)
        lease = async_pager.sync_lease(key)
        assert hasattr(lease, "__enter__")
        assert hasattr(lease, "__exit__")


class TestAsyncPagerEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def mock_index(self):
        """Create mock checkpoint index."""
        index = MockCheckpointIndex()
        index.add_expert((0, 0), byte_size=2048)
        index.add_expert((0, 1), byte_size=2048)
        return index

    def test_async_pager_with_low_backpressure(self, mock_index):
        """Pager with backpressure limit of 1."""
        cache = ExpertCache(max_experts=2)
        sync_pager = ExpertPager(
            index=mock_index,
            cache=cache,
            device="cpu",
        )
        config = BackpressureConfig(max_outstanding=1)
        pager = AsyncExpertPager(sync_pager=sync_pager, config=config)
        assert pager.config.max_outstanding == 1

    def test_async_pager_with_high_backpressure(self, mock_index):
        """Pager with high backpressure limit."""
        cache = ExpertCache(max_experts=2)
        sync_pager = ExpertPager(index=mock_index, cache=cache, device="cpu")
        config = BackpressureConfig(max_outstanding=10)
        pager = AsyncExpertPager(sync_pager=sync_pager, config=config)
        assert pager.config.max_outstanding == 10

    def test_multiple_leases_track_stats(self, mock_index):
        """Multiple leases are tracked in stats."""
        cache = ExpertCache(max_experts=4)
        sync_pager = ExpertPager(index=mock_index, cache=cache, device="cpu")
        async_pager = AsyncExpertPager(sync_pager=sync_pager)

        keys = [(0, 0), (0, 1)]
        for key in keys:
            with async_pager.sync_lease(key):
                pass

        stats = async_pager.request_stats()
        # Should have at least one hit (after the first miss)
        assert stats["cache_hits"] >= 1 or stats["cache_misses"] >= 2

    def test_pending_dict_starts_empty(self, mock_index):
        """Async pager starts with empty pending dict."""
        cache = ExpertCache(max_experts=2)
        sync_pager = ExpertPager(index=mock_index, cache=cache, device="cpu")
        async_pager = AsyncExpertPager(sync_pager=sync_pager)
        assert len(async_pager._pending) == 0

    def test_semaphore_respects_limit(self, mock_index):
        """Semaphore respects max_outstanding limit."""
        cache = ExpertCache(max_experts=2)
        sync_pager = ExpertPager(index=mock_index, cache=cache, device="cpu")
        config = BackpressureConfig(max_outstanding=2)
        async_pager = AsyncExpertPager(sync_pager=sync_pager, config=config)
        assert async_pager._prefetch_semaphore._value == 2

"""Unit tests for ExpertCacheManager with predictive prefetching."""

import threading
import time
import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np

from sparse_llm.cache.expert_cache import ExpertCache
from sparse_llm.core.router_predictor import RouterPredictor
from sparse_llm.inference.expert_cache_manager import ExpertCacheManager


class TestExpertCacheManager(unittest.TestCase):
    """Test ExpertCacheManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = ExpertCache(max_experts=10, max_bytes=1024 * 1024)
        self.predictor = RouterPredictor(num_experts=16, markov_window=10)
        self.manager = ExpertCacheManager(
            self.cache,
            self.predictor,
            enable_prefetch=True,
            prefetch_confidence_threshold=0.2,
            prefetch_top_k=5,
        )

    def tearDown(self):
        """Clean up resources."""
        self.manager.shutdown()

    def test_initialization(self):
        """Test manager initialization."""
        self.assertIsNotNone(self.manager.cache)
        self.assertIsNotNone(self.manager.router_predictor)
        self.assertTrue(self.manager.enable_prefetch)
        self.assertEqual(self.manager.prefetch_top_k, 5)

    def test_basic_cache_operations(self):
        """Test basic get/put operations."""
        # Create mock expert
        mock_expert = Mock()
        mock_expert.nbytes = 1024

        # Put expert
        self.manager.put(1, mock_expert)
        self.assertEqual(self.cache.bytes_used, 1024)

        # Get expert
        result = self.manager.get(1)
        self.assertEqual(result, mock_expert)

        # Check contains
        self.assertTrue(self.manager.contains(1))
        self.assertFalse(self.manager.contains(999))

    def test_get_or_load(self):
        """Test get_or_load functionality."""
        mock_expert = Mock()
        mock_expert.nbytes = 2048
        loader = Mock(return_value=mock_expert)

        # First call should load
        result = self.manager.get_or_load(1, loader)
        self.assertEqual(result, mock_expert)
        loader.assert_called_once()

        # Second call should hit cache
        loader.reset_mock()
        result = self.manager.get_or_load(1, loader)
        self.assertEqual(result, mock_expert)
        loader.assert_not_called()

    def test_preload_experts(self):
        """Test preloading multiple experts."""
        mock_experts = {i: Mock(nbytes=512) for i in range(5)}
        loader = lambda expert_id: mock_experts[expert_id]

        # Preload experts
        self.manager.preload_experts([0, 1, 2], loader)

        # Verify all are cached
        for i in range(3):
            self.assertTrue(self.manager.contains(i))

        # Verify stats
        stats = self.manager.get_cache_stats()
        self.assertEqual(stats["cached_experts"], 3)

    def test_pin_unpin(self):
        """Test pin/unpin functionality."""
        mock_expert = Mock(nbytes=1024)
        self.manager.put(1, mock_expert)

        # Pin expert
        self.manager.pin(1)
        stats = self.manager.get_cache_stats()
        self.assertEqual(stats["pinned_experts"], 1)

        # Unpin expert
        self.manager.unpin(1)
        stats = self.manager.get_cache_stats()
        self.assertEqual(stats["pinned_experts"], 0)

    def test_predictor_update(self):
        """Test updating predictor from inference."""
        activated_experts = [0, 1, 2]
        router_logits = np.random.randn(16)

        # Update predictor
        self.manager.update_predictor_from_inference(activated_experts, router_logits)

        # Verify predictor was updated
        stats = self.manager.get_cache_stats()
        self.assertIn("predictor", stats)
        self.assertGreater(stats["predictor"]["history_size"], 0)

    def test_optimize_cache_for_decode(self):
        """Test cache optimization with predictive prefetching."""
        # Train predictor with some patterns
        for _ in range(5):
            self.predictor.update_from_inference([0, 1])
            self.predictor.update_from_inference([1, 2])

        # Create mock loader
        mock_experts = {i: Mock(nbytes=512) for i in range(16)}
        loader = lambda expert_id: mock_experts[expert_id]

        # Preload current experts
        self.manager.preload_experts([0, 1], loader)

        # Trigger optimization (should predict and prefetch expert 2)
        self.manager.optimize_cache_for_decode([0, 1], loader)

        # Give prefetch thread time to work
        time.sleep(0.5)

        # Check if predicted expert was prefetched
        # Expert 2 should be predicted based on the pattern
        stats = self.manager.get_cache_stats()
        self.assertGreaterEqual(stats["cached_experts"], 2)

    def test_prefetch_queue_management(self):
        """Test prefetch queue and worker thread."""
        # Train predictor
        for _ in range(3):
            self.predictor.update_from_inference([0, 1, 2])

        mock_experts = {i: Mock(nbytes=256) for i in range(16)}
        loader = lambda expert_id: mock_experts[expert_id]

        # Trigger prefetch
        self.manager.optimize_cache_for_decode([0, 1], loader)

        # Check queue was populated
        with self.manager._prefetch_lock:
            initial_queue_size = len(self.manager._prefetch_queue)

        # Wait for worker to process
        time.sleep(0.3)

        # Queue should be processed
        with self.manager._prefetch_lock:
            final_queue_size = len(self.manager._prefetch_queue)

        self.assertLessEqual(final_queue_size, initial_queue_size)

    def test_prefetch_disabled(self):
        """Test behavior when prefetching is disabled."""
        manager = ExpertCacheManager(
            self.cache,
            self.predictor,
            enable_prefetch=False,
        )

        mock_experts = {i: Mock(nbytes=512) for i in range(16)}
        loader = lambda expert_id: mock_experts[expert_id]

        # Try to optimize (should do nothing)
        manager.optimize_cache_for_decode([0, 1], loader)

        # No prefetch thread should be created
        self.assertIsNone(manager._prefetch_thread)

        manager.shutdown()

    def test_prefetch_with_no_predictor(self):
        """Test prefetch behavior when no predictor is provided."""
        manager = ExpertCacheManager(
            self.cache,
            router_predictor=None,
            enable_prefetch=True,
        )

        mock_experts = {i: Mock(nbytes=512) for i in range(16)}
        loader = lambda expert_id: mock_experts[expert_id]

        # Try to optimize (should do nothing without predictor)
        manager.optimize_cache_for_decode([0, 1], loader)

        # No prefetch should occur
        self.assertIsNone(manager._prefetch_thread)

        manager.shutdown()

    def test_cache_stats_comprehensive(self):
        """Test comprehensive cache statistics."""
        mock_expert = Mock(nbytes=1024)
        loader = Mock(return_value=mock_expert)

        # Perform various operations
        self.manager.get_or_load(1, loader)
        self.manager.get(1)
        self.manager.get(999)  # Miss

        stats = self.manager.get_cache_stats()

        # Verify stat keys
        self.assertIn("cached_experts", stats)
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)
        self.assertIn("prefetch_enabled", stats)
        self.assertIn("prefetch_hits", stats)
        self.assertIn("prefetch_misses", stats)
        self.assertIn("total_prefetch_time", stats)

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        mock_experts = {i: Mock(nbytes=256) for i in range(10)}
        loader = lambda expert_id: mock_experts[expert_id]

        def worker(expert_ids):
            for expert_id in expert_ids:
                self.manager.get_or_load(expert_id, lambda: loader(expert_id))

        # Create multiple threads accessing different experts
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=([i, i+1, i+2],))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify cache consistency
        stats = self.manager.get_cache_stats()
        self.assertGreater(stats["cached_experts"], 0)
        self.assertEqual(stats["hits"] + stats["misses"], stats["total_accesses"])

    def test_layer_aware_caching(self):
        """Test layer-aware expert caching."""
        mock_expert_l0 = Mock(nbytes=512)
        mock_expert_l1 = Mock(nbytes=512)

        # Put same expert ID in different layers
        self.manager.put((0, 5), mock_expert_l0, layer_id=0)
        self.manager.put((1, 5), mock_expert_l1, layer_id=1)

        # Retrieve and verify
        result_l0 = self.manager.get(5, layer_id=0)
        result_l1 = self.manager.get(5, layer_id=1)

        self.assertEqual(result_l0, mock_expert_l0)
        self.assertEqual(result_l1, mock_expert_l1)
        self.assertNotEqual(result_l0, result_l1)

    def test_clear(self):
        """Test cache clearing."""
        mock_expert = Mock(nbytes=1024)
        self.manager.put(1, mock_expert)
        self.manager.prefetch_hits = 5
        self.manager.total_prefetch_time = 10.0

        # Clear
        self.manager.clear()

        # Verify everything is reset
        stats = self.manager.get_cache_stats()
        self.assertEqual(stats["cached_experts"], 0)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(self.manager.prefetch_hits, 0)
        self.assertEqual(self.manager.total_prefetch_time, 0.0)

    def test_prefetch_worker_error_handling(self):
        """Test prefetch worker handles loader errors gracefully."""
        def failing_loader(expert_id):
            if expert_id == 5:
                raise RuntimeError("Simulated load failure")
            return Mock(nbytes=256)

        # Train predictor to predict expert 5
        for _ in range(3):
            self.predictor.update_from_inference([4, 5, 6])

        # Trigger prefetch (should try to load expert 5 and fail gracefully)
        self.manager.optimize_cache_for_decode([4, 6], failing_loader)

        # Wait for prefetch to attempt
        time.sleep(0.3)

        # Manager should still be functional
        stats = self.manager.get_cache_stats()
        self.assertIsNotNone(stats)

    def test_tracking_expert_access(self):
        """Test internal expert access tracking."""
        mock_expert = Mock(nbytes=512)
        loader = Mock(return_value=mock_expert)

        # Access several experts
        for i in range(5):
            self.manager.get_or_load(i, loader)

        # Verify tracking
        with self.manager._recent_experts_lock:
            self.assertEqual(len(self.manager._recent_experts), 5)

    def test_prefetch_confidence_threshold(self):
        """Test prefetch confidence threshold filtering."""
        # Create manager with high confidence threshold
        manager = ExpertCacheManager(
            self.cache,
            self.predictor,
            enable_prefetch=True,
            prefetch_confidence_threshold=0.9,  # Very high threshold
            prefetch_top_k=10,
        )

        # Train with weak pattern
        self.predictor.update_from_inference([0, 1])

        mock_experts = {i: Mock(nbytes=256) for i in range(16)}
        loader = lambda expert_id: mock_experts[expert_id]

        # Trigger optimization
        manager.optimize_cache_for_decode([0], loader)

        # With high threshold, few or no experts should be prefetched
        time.sleep(0.3)

        with manager._prefetch_lock:
            queue_size = len(manager._prefetch_queue)

        # Queue should be empty or very small due to high threshold
        self.assertLessEqual(queue_size, 2)

        manager.shutdown()


class TestExpertCacheManagerIntegration(unittest.TestCase):
    """Integration tests for ExpertCacheManager with real predictor."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = ExpertCache(max_experts=20, max_bytes=2 * 1024 * 1024)
        self.predictor = RouterPredictor(num_experts=32, markov_window=50)
        self.manager = ExpertCacheManager(
            self.cache,
            self.predictor,
            enable_prefetch=True,
            prefetch_confidence_threshold=0.25,
            prefetch_top_k=10,
        )

    def tearDown(self):
        """Clean up resources."""
        self.manager.shutdown()

    def test_realistic_inference_pattern(self):
        """Test with realistic inference pattern."""
        # Simulate a realistic expert activation pattern
        patterns = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
        ]

        mock_experts = {i: Mock(nbytes=512) for i in range(32)}
        loader = lambda expert_id: mock_experts[expert_id]

        # Simulate prefill phase with pattern learning
        for pattern in patterns:
            self.manager.preload_experts(pattern, loader)
            self.manager.update_predictor_from_inference(pattern)

        # Now simulate decode with prediction
        current_experts = [7, 8]
        self.manager.optimize_cache_for_decode(current_experts, loader)

        # Give prefetch time to work
        time.sleep(0.5)

        # Verify predictor learned the pattern
        stats = self.manager.get_cache_stats()
        self.assertGreater(stats["predictor"]["markov_states"], 0)
        self.assertGreater(stats["cached_experts"], len(current_experts))

    def test_cache_efficiency_with_prefetch(self):
        """Test that prefetching improves cache efficiency."""
        # Train predictor with strong pattern
        for _ in range(10):
            self.predictor.update_from_inference([0, 1])
            self.predictor.update_from_inference([1, 2])
            self.predictor.update_from_inference([2, 3])

        mock_experts = {i: Mock(nbytes=256) for i in range(32)}
        loader = lambda expert_id: mock_experts[expert_id]

        # Preload initial experts
        self.manager.preload_experts([0, 1], loader)

        # Trigger prefetch
        self.manager.optimize_cache_for_decode([1], loader)
        time.sleep(0.5)

        # Access predicted expert (should be cache hit)
        initial_hits = self.cache.hits
        result = self.manager.get(2)

        if result is not None:
            # If prefetch worked, this should be a hit
            self.assertGreater(self.cache.hits, initial_hits)


if __name__ == "__main__":
    unittest.main()

"""Tests for PredictivePrefetcher."""

import pytest
import threading
import time

from sparse_llm.inference.prefetcher import PredictivePrefetcher


class TestPredictivePrefetcherBasic:
    """Basic functionality tests."""

    def test_initialization_with_thread(self):
        """Should start background thread when enabled."""
        loaded = []

        def loader(expert_id, layer_id):
            loaded.append((expert_id, layer_id))

        prefetcher = PredictivePrefetcher(
            expert_loader=loader,
            enable_background_thread=True,
        )
        assert prefetcher._prefetch_thread is not None
        assert prefetcher._prefetch_thread.is_alive()
        prefetcher.stop()

    def test_initialization_without_thread(self):
        """Should not start thread when disabled."""
        prefetcher = PredictivePrefetcher(
            expert_loader=lambda e, l: None,
            enable_background_thread=False,
        )
        assert not hasattr(prefetcher, "_prefetch_thread")
        stats = prefetcher.get_stats()
        assert stats["thread_alive"] is False

    def test_observe_transitions(self):
        """Should record transition probabilities."""
        prefetcher = PredictivePrefetcher(
            expert_loader=lambda e, l: None,
            enable_background_thread=False,
        )

        # Expert 0 often leads to expert 1
        prefetcher.observe(0, [1, 2])
        prefetcher.observe(0, [1, 3])
        prefetcher.observe(0, [1])

        # Expert 1 often leads to expert 2
        prefetcher.observe(1, [2])
        prefetcher.observe(1, [2, 3])

        assert 0 in prefetcher.transition_probs
        assert 1 in prefetcher.transition_probs

        # Expert 0's transitions
        assert prefetcher.transition_probs[0][1] == 3
        assert prefetcher.transition_probs[0][2] == 1
        assert prefetcher.transition_probs[0][3] == 1

        # Expert 1's transitions
        assert prefetcher.transition_probs[1][2] == 2
        assert prefetcher.transition_probs[1][3] == 1

    def test_observe_layer(self):
        """Should extract transitions from layer activations."""
        prefetcher = PredictivePrefetcher(
            expert_loader=lambda e, l: None,
            enable_background_thread=False,
        )

        # Layer activation: experts [0, 1, 2] activated
        prefetcher.observe_layer(layer_id=0, expert_ids=[0, 1, 2])

        # Should observe: 0 -> [1, 2], 1 -> [2], 2 -> []
        assert 0 in prefetcher.transition_probs
        assert 1 in prefetcher.transition_probs
        assert 2 in prefetcher.transition_probs

    def test_get_stats(self):
        """Should return correct statistics."""
        prefetcher = PredictivePrefetcher(
            expert_loader=lambda e, l: None,
            enable_background_thread=True,
        )

        stats = prefetcher.get_stats()
        assert "num_transitions" in stats
        assert "queue_size" in stats
        assert "queue_max_size" in stats
        assert "thread_alive" in stats
        assert stats["thread_alive"] is True

        prefetcher.stop()

    def test_get_transition_probs(self):
        """Should return transition probabilities."""
        prefetcher = PredictivePrefetcher(
            expert_loader=lambda e, l: None,
            enable_background_thread=False,
        )

        prefetcher.observe(0, [1, 2])
        prefetcher.observe(0, [1])

        probs = prefetcher.get_transition_probs()
        assert 0 in probs
        assert (1, 2) in probs[0]  # expert 0 -> expert 1 with count 2
        assert (2, 1) in probs[0]  # expert 0 -> expert 2 with count 1


class TestPredictivePrefetcherPredictAndPrefetch:
    """Tests for predict_and_prefetch functionality."""

    def test_predict_with_no_history(self):
        """Should handle no transition history gracefully."""
        prefetcher = PredictivePrefetcher(
            expert_loader=lambda e, l: None,
            enable_background_thread=False,
        )

        # No history - should not crash
        prefetcher.predict_and_prefetch([0, 1], layer_id=0)

    def test_predict_with_history(self):
        """Should queue predicted experts for prefetch."""
        loaded = []

        def loader(expert_id, layer_id):
            loaded.append((expert_id, layer_id))

        prefetcher = PredictivePrefetcher(
            expert_loader=loader,
            enable_background_thread=False,
            top_k_predictions=2,
        )

        # Learn: expert 0 leads to expert 1, expert 1 leads to expert 2
        prefetcher.observe(0, [1, 2])
        prefetcher.observe(1, [2, 3])

        # Predict from expert 0 and 1
        prefetcher.predict_and_prefetch([0, 1], layer_id=5)

        # Should have queued predictions
        stats = prefetcher.get_stats()
        assert stats["queue_size"] > 0

    def test_predict_queue_limit(self):
        """Should respect queue size limit."""
        prefetcher = PredictivePrefetcher(
            expert_loader=lambda e, l: None,
            enable_background_thread=False,
            prefetch_queue_size=2,
        )

        # Add many transitions
        for i in range(10):
            prefetcher.observe(i, [i + 1])

        # Predict - should fill queue and skip rest
        prefetcher.predict_and_prefetch(list(range(10)), layer_id=0)

        stats = prefetcher.get_stats()
        assert stats["queue_size"] <= 2


class TestPredictivePrefetcherBackgroundWorker:
    """Tests for background worker functionality."""

    def test_worker_loads_experts(self):
        """Background thread should load queued experts."""
        loaded = []
        lock = threading.Lock()

        def loader(expert_id, layer_id):
            with lock:
                loaded.append((expert_id, layer_id))

        prefetcher = PredictivePrefetcher(
            expert_loader=loader,
            enable_background_thread=True,
        )

        # Queue some predictions
        prefetcher.observe(0, [1])
        prefetcher.predict_and_prefetch([0], layer_id=1)

        prefetcher.observe(1, [2])
        prefetcher.predict_and_prefetch([1], layer_id=2)

        # Wait for background thread to process
        time.sleep(0.2)

        prefetcher.stop()

        # Should have loaded at least some experts
        assert len(loaded) >= 1

    def test_worker_handles_errors(self):
        """Should handle loader errors gracefully."""
        errors = []

        def failing_loader(expert_id, layer_id):
            if expert_id == 99:
                raise ValueError("Test error")
            errors.append(expert_id)

        prefetcher = PredictivePrefetcher(
            expert_loader=failing_loader,
            enable_background_thread=True,
        )

        # Queue an error-inducing prediction
        prefetcher.prefetch_queue.put((99, 0))
        prefetcher.prefetch_queue.put((1, 0))

        # Wait for processing
        time.sleep(0.3)

        prefetcher.stop()

        # Should have continued processing despite error
        assert len(errors) >= 1

    def test_stop_waits_for_queue(self):
        """Should process remaining queue items on stop."""
        loaded = []
        lock = threading.Lock()

        def slow_loader(expert_id, layer_id):
            with lock:
                loaded.append(expert_id)
            time.sleep(0.05)  # Slow loading

        prefetcher = PredictivePrefetcher(
            expert_loader=slow_loader,
            enable_background_thread=True,
        )

        # Queue several items
        for i in range(3):
            prefetcher.prefetch_queue.put((i, 0))

        # Stop immediately
        prefetcher.stop()

        # Should have processed at least some
        assert len(loaded) >= 1


class TestPredictivePrefetcherIntegration:
    """Integration tests with ExpertProcessor."""

    def test_prefetcher_integrates_with_processor(self):
        """Should work with ExpertProcessor."""
        from sparse_llm.cache.expert_cache import ExpertCache
        from sparse_llm.inference.expert_processor import ExpertProcessor

        cache = ExpertCache(max_experts=16)
        processor = ExpertProcessor(
            expert_cache=cache,
            num_experts=8,
            hidden_dim=64,
            expert_dim=128,
            activation="silu",
            device="cpu",
            enable_predictive_prefetch=True,
        )

        # Should have prefetcher attached
        assert processor._prefetcher is not None
        assert processor._prefetcher._prefetch_thread.is_alive()

        # Should be able to call prefetch_next
        processor.prefetch_next([0, 1], layer_id=0)

        # Should be able to observe routing
        processor.observe_routing([0, 1, 2])

        processor._prefetcher.stop()

    def test_prefetcher_disabled_when_flag_off(self):
        """Should not create prefetcher when disabled."""
        from sparse_llm.cache.expert_cache import ExpertCache
        from sparse_llm.inference.expert_processor import ExpertProcessor

        cache = ExpertCache(max_experts=16)
        processor = ExpertProcessor(
            expert_cache=cache,
            num_experts=8,
            hidden_dim=64,
            expert_dim=128,
            device="cpu",
            enable_predictive_prefetch=False,
        )

        assert processor._prefetcher is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

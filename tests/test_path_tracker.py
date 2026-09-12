"""Tests for ExpertPathTracker."""

import pytest
import torch
from sparse_llm.inference.path_tracker import ExpertPathTracker


class TestExpertPathTracker:
    """Test ExpertPathTracker functionality."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance for tests."""
        return ExpertPathTracker(num_layers=4, top_k=2)

    def test_init(self, tracker):
        """Should initialize with correct attributes."""
        assert tracker.num_layers == 4
        assert tracker.top_k == 2
        assert tracker._path_history == []
        assert tracker._expert_frequency == {}
        assert tracker._transitions == {}

    def test_record_path_returns_fingerprint(self, tracker):
        """Should return a fingerprint for routing decisions."""
        routing = torch.randint(0, 8, (1, 2, 4, 2))  # [batch=1, seq=2, layers=4, top_k=2]
        fp = tracker.record_path(routing)
        assert isinstance(fp, int)

    def test_record_path_updates_history(self, tracker):
        """Should append fingerprint to history."""
        routing = torch.randint(0, 8, (1, 2, 4, 2))
        fp1 = tracker.record_path(routing)
        fp2 = tracker.record_path(routing)
        assert len(tracker._path_history) == 2
        assert tracker._path_history[0] == fp1
        assert tracker._path_history[1] == fp2

    def test_record_path_updates_frequency(self, tracker):
        """Should track expert frequency."""
        # Shape: [batch=1, seq=1, layers=4, top_k=2]
        routing = torch.tensor([[[[0], [1], [2], [3]]]])  # Each layer has one expert
        routing = routing.repeat(1, 1, 1, 2)  # Now [1, 1, 4, 2] with duplicated experts
        tracker.record_path(routing)
        # Each expert appears twice (duplicated across top_k)
        for e in range(4):
            assert tracker._expert_frequency[e] == 2

    def test_record_path_updates_transitions(self, tracker):
        """Should track expert transitions between layers."""
        routing = torch.tensor([[[0, 1], [2, 3], [0, 2], [1, 3]]])
        routing = routing.unsqueeze(0)  # [1, 1, 4, 2]
        tracker.record_path(routing)
        # Should have transitions from layer 0 experts
        assert 0 in tracker._transitions
        assert 1 in tracker._transitions

    def test_compute_fingerprint_deterministic(self, tracker):
        """Should produce same fingerprint for same routing."""
        routing = torch.randint(0, 8, (1, 2, 4, 2))
        fp1 = tracker._compute_fingerprint(routing)
        fp2 = tracker._compute_fingerprint(routing)
        assert fp1 == fp2

    def test_compute_fingerprint_differs_for_different_routing(self, tracker):
        """Should produce different fingerprints for different routing."""
        routing1 = torch.tensor([[[[0, 1], [2, 3], [4, 5], [6, 7]]]])
        routing2 = torch.tensor([[[[1, 0], [3, 2], [5, 4], [7, 6]]]])
        fp1 = tracker._compute_fingerprint(routing1)
        fp2 = tracker._compute_fingerprint(routing2)
        assert fp1 != fp2

    def test_predict_hot_experts(self, tracker):
        """Should return top N frequent experts."""
        # Shape: [batch=1, seq=1, layers=4, top_k=2]
        # Expert 3 appears 2 times (1 layer * 2 top_k), expert 0 appears 6 times (3 layers * 2)
        routing = torch.tensor([[[[0, 0], [0, 0], [0, 0], [3, 3]]]])
        tracker.record_path(routing)
        hot = tracker.predict_hot_experts(top_n=2)
        assert hot[0] == 0  # Most frequent should be first
        assert hot[1] == 3  # Second most

    def test_predict_hot_experts_default_top_8(self, tracker):
        """Should default to top 8 experts."""
        routing = torch.randint(0, 16, (1, 2, 4, 2))
        tracker.record_path(routing)
        hot = tracker.predict_hot_experts()
        assert len(hot) <= 8

    def test_get_next_experts(self, tracker):
        """Should predict next experts based on transitions."""
        # Expert 0 always followed by expert 2
        routing = torch.tensor([[[[0, 1], [2, 3], [0, 1], [2, 3]]]])
        tracker.record_path(routing)
        next_experts = tracker.get_next_experts([0])
        assert 2 in next_experts

    def test_get_next_experts_empty_input(self, tracker):
        """Should handle empty expert list."""
        next_experts = tracker.get_next_experts([])
        assert next_experts == []

    def test_get_next_experts_unknown_expert(self, tracker):
        """Should handle unknown experts gracefully."""
        routing = torch.randint(0, 8, (1, 2, 4, 2))
        tracker.record_path(routing)
        next_experts = tracker.get_next_experts([999])  # Unknown expert
        assert next_experts == []

    def test_reset_clears_history(self, tracker):
        """Should clear path history but keep patterns."""
        routing = torch.randint(0, 8, (1, 2, 4, 2))
        tracker.record_path(routing)
        tracker.record_path(routing)
        assert len(tracker._path_history) == 2

        tracker.reset()

        assert tracker._path_history == []
        # Patterns should be preserved
        assert len(tracker._expert_frequency) > 0

    def test_empty_routing(self, tracker):
        """Should handle empty routing tensor."""
        routing = torch.randint(0, 8, (0, 2, 4, 2))
        fp = tracker.record_path(routing)
        assert isinstance(fp, int)
        assert len(tracker._path_history) == 1

    def test_large_batch(self, tracker):
        """Should handle large batch sizes."""
        routing = torch.randint(0, 16, (32, 64, 4, 2))  # Large batch
        fp = tracker.record_path(routing)
        assert isinstance(fp, int)
        assert len(tracker._path_history) == 1

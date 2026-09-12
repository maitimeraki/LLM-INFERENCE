"""Tests for ExpertProcessor grouping functionality."""

import pytest
import torch

from sparse_llm.cache.expert_cache import ExpertCache
from sparse_llm.inference.expert_processor import ExpertProcessor, ExpertFFN


class TestExpertProcessorGrouping:
    """Test expert grouping by fingerprint."""

    @pytest.fixture
    def processor(self):
        """Create processor instance for tests."""
        cache = ExpertCache(max_experts=16)
        return ExpertProcessor(
            expert_cache=cache,
            num_experts=8,
            hidden_dim=64,
            expert_dim=128,
            device="cpu",
        )

    @pytest.fixture
    def mock_experts(self):
        """Create mock experts."""
        experts = {}
        for i in range(8):
            experts[i] = ExpertFFN(hidden_dim=64, expert_dim=128)
        return experts

    def test_compute_batch_fingerprints_shape(self, processor):
        """Should return tensor with correct shape."""
        batch_size, seq_len, top_k = 2, 4, 2
        routing = torch.randint(0, 8, (batch_size, seq_len, top_k))
        fps = processor._compute_batch_fingerprints(routing)
        assert fps.shape == (batch_size, seq_len)

    def test_compute_batch_fingerprints_deterministic(self, processor):
        """Same routing should produce same fingerprints."""
        routing = torch.randint(0, 8, (2, 4, 2))
        fps1 = processor._compute_batch_fingerprints(routing)
        fps2 = processor._compute_batch_fingerprints(routing)
        assert torch.equal(fps1, fps2)

    def test_compute_batch_fingerprints_different_for_different_routing(self, processor):
        """Different routing should produce different fingerprints."""
        routing1 = torch.tensor([[[0, 1], [2, 3]]])
        routing2 = torch.tensor([[[1, 0], [3, 2]]])
        fps1 = processor._compute_batch_fingerprints(routing1)
        fps2 = processor._compute_batch_fingerprints(routing2)
        assert not torch.equal(fps1, fps2)

    def test_group_by_fingerprint_returns_dict(self, processor):
        """Should return dictionary with fingerprints as keys."""
        batch_size, seq_len = 2, 4
        hidden = torch.randn(batch_size, seq_len, 64)
        routing = torch.randint(0, 8, (batch_size, seq_len, 2))
        fps = processor._compute_batch_fingerprints(routing)
        groups = processor._group_by_fingerprint(hidden, routing, fps)
        assert isinstance(groups, dict)
        assert all(isinstance(k, int) for k in groups.keys())

    def test_group_by_fingerprint_has_required_keys(self, processor):
        """Each group should have indices, expert_ids, hidden."""
        batch_size, seq_len = 2, 4
        hidden = torch.randn(batch_size, seq_len, 64)
        routing = torch.randint(0, 8, (batch_size, seq_len, 2))
        fps = processor._compute_batch_fingerprints(routing)
        groups = processor._group_by_fingerprint(hidden, routing, fps)
        for fp, group in groups.items():
            assert 'indices' in group
            assert 'expert_ids' in group
            assert 'hidden' in group
            assert isinstance(group['indices'], torch.Tensor)
            assert isinstance(group['expert_ids'], set)

    def test_group_by_fingerprint_all_tokens_covered(self, processor):
        """All tokens should appear in some group."""
        batch_size, seq_len = 2, 4
        hidden = torch.randn(batch_size, seq_len, 64)
        routing = torch.randint(0, 8, (batch_size, seq_len, 2))
        fps = processor._compute_batch_fingerprints(routing)
        groups = processor._group_by_fingerprint(hidden, routing, fps)
        total_indices = sum(len(g['indices']) for g in groups.values())
        assert total_indices == batch_size * seq_len

    def test_group_by_fingerprint_same_fingerprint_same_experts(self, processor):
        """Tokens with same fingerprint should have same expert IDs."""
        batch_size, seq_len = 4, 2
        hidden = torch.randn(batch_size, seq_len, 64)
        # Force same routing for all tokens
        routing = torch.zeros(batch_size, seq_len, 2, dtype=torch.long)
        routing[:, :, 0] = 2
        routing[:, :, 1] = 5
        fps = processor._compute_batch_fingerprints(routing)
        groups = processor._group_by_fingerprint(hidden, routing, fps)
        # All tokens should be in one group
        assert len(groups) == 1
        group = list(groups.values())[0]
        assert group['expert_ids'] == {2, 5}

    def test_process_group_returns_correct_shape(self, processor, mock_experts):
        """Should return tensor with correct shape."""
        hidden = torch.randn(4, 64)
        routing = torch.tensor([[0, 1], [0, 1], [2, 3], [2, 3]])
        weights = torch.softmax(torch.randn(4, 2), dim=-1)
        expert_modules = {i: mock_experts[i] for i in range(4)}
        output = processor._process_group(hidden, routing, weights, expert_modules)
        assert output.shape == hidden.shape

    def test_process_group_single_expert(self, processor, mock_experts):
        """Should handle single expert correctly."""
        hidden = torch.randn(2, 64)
        routing = torch.tensor([[0, 0], [0, 0]])  # Same expert twice
        weights = torch.ones(2, 2)
        expert_modules = {0: mock_experts[0]}
        output = processor._process_group(hidden, routing, weights, expert_modules)
        assert output.shape == hidden.shape

    def test_process_group_missing_expert(self, processor, mock_experts):
        """Should handle missing experts gracefully."""
        hidden = torch.randn(2, 64)
        routing = torch.tensor([[0, 1], [0, 1]])
        weights = torch.ones(2, 2)
        # Only load expert 0
        expert_modules = {0: mock_experts[0]}
        output = processor._process_group(hidden, routing, weights, expert_modules)
        # Should still produce output
        assert output.shape == hidden.shape


class TestExpertProcessorGroupedBatchProcessing:
    """Test process_batch with grouping."""

    @pytest.fixture
    def processor(self):
        """Create processor instance for tests."""
        cache = ExpertCache(max_experts=16)
        return ExpertProcessor(
            expert_cache=cache,
            num_experts=8,
            hidden_dim=64,
            expert_dim=128,
            device="cpu",
        )

    @pytest.fixture
    def mock_experts(self):
        """Create mock experts."""
        experts = {}
        for i in range(8):
            experts[i] = ExpertFFN(hidden_dim=64, expert_dim=128)
        return experts

    def test_process_batch_output_shape(self, processor, mock_experts):
        """Output should match input shape."""
        batch_size, seq_len = 2, 4
        hidden = torch.randn(batch_size, seq_len, 64)
        routing = torch.randint(0, 8, (batch_size, seq_len, 2))
        weights = torch.softmax(torch.randn(batch_size, seq_len, 2), dim=-1)
        loader = lambda eid: mock_experts[eid]
        output = processor.process_batch(hidden, routing, weights, expert_loader=loader)
        assert output.shape == hidden.shape

    def test_process_batch_with_identical_routing(self, processor, mock_experts):
        """All tokens same routing should produce one group."""
        batch_size, seq_len = 3, 4
        hidden = torch.randn(batch_size, seq_len, 64)
        # All tokens go to experts 1 and 2
        routing = torch.full((batch_size, seq_len, 2), 1)
        routing[:, :, 1] = 2
        weights = torch.softmax(torch.randn(batch_size, seq_len, 2), dim=-1)
        loader = lambda eid: mock_experts[eid]
        output = processor.process_batch(hidden, routing, weights, expert_loader=loader)
        assert output.shape == hidden.shape

    def test_process_batch_with_different_routing(self, processor, mock_experts):
        """Tokens with different routing should go to different groups."""
        batch_size, seq_len = 4, 2
        hidden = torch.randn(batch_size, seq_len, 64)
        routing = torch.randint(0, 8, (batch_size, seq_len, 2))
        weights = torch.softmax(torch.randn(batch_size, seq_len, 2), dim=-1)
        loader = lambda eid: mock_experts[eid]
        output = processor.process_batch(hidden, routing, weights, expert_loader=loader)
        assert output.shape == hidden.shape

    def test_process_batch_without_loader(self, processor, mock_experts):
        """Should work without loader when experts are preloaded."""
        cache = ExpertCache(max_experts=16)
        # Preload experts
        for i in range(8):
            cache.put(i, mock_experts[i])
        processor = ExpertProcessor(
            expert_cache=cache,
            num_experts=8,
            hidden_dim=64,
            expert_dim=128,
            device="cpu",
        )
        batch_size, seq_len = 2, 4
        hidden = torch.randn(batch_size, seq_len, 64)
        routing = torch.randint(0, 8, (batch_size, seq_len, 2))
        weights = torch.softmax(torch.randn(batch_size, seq_len, 2), dim=-1)
        output = processor.process_batch(hidden, routing, weights)
        assert output.shape == hidden.shape

    def test_process_batch_tracks_activated_experts(self, processor, mock_experts):
        """Should track which experts were activated."""
        batch_size, seq_len = 2, 4
        hidden = torch.randn(batch_size, seq_len, 64)
        routing = torch.randint(0, 8, (batch_size, seq_len, 2))
        weights = torch.softmax(torch.randn(batch_size, seq_len, 2), dim=-1)
        loader = lambda eid: mock_experts[eid]
        processor.process_batch(hidden, routing, weights, expert_loader=loader)
        assert len(processor._last_activated_experts) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

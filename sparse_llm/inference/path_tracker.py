"""Expert Path Tracker for expert-centric batching.

Tracks routing paths to enable grouping tokens by their full expert usage,
reducing expert loads from "per token" to "per unique path".
"""

from collections import Counter
from typing import Optional
import torch


class ExpertPathTracker:
    """
    Track routing paths to enable expert-centric batching.

    For each token, record:
    - path_fingerprint: hash of (layer_0_experts, layer_1_experts, ...)
    - routing_weights: per-layer routing probabilities
    """

    def __init__(self, num_layers: int, top_k: int):
        self.num_layers = num_layers
        self.top_k = top_k
        self._path_history: list[int] = []  # fingerprints
        self._expert_frequency: Counter = Counter()
        self._transitions: dict[int, Counter] = {}  # expert -> next_experts

    def record_path(self, routing_decisions: torch.Tensor) -> int:
        """
        Record routing path and return fingerprint for grouping.

        Args:
            routing_decisions: [batch, seq, num_layers, top_k] expert indices

        Returns:
            fingerprint: int hash of the routing path
        """
        # Compute fingerprint from routing decisions
        fingerprint = self._compute_fingerprint(routing_decisions)
        self._path_history.append(fingerprint)

        # Update frequency - count all expert appearances
        flat_experts = routing_decisions.flatten().tolist()
        for expert_id in flat_experts:
            self._expert_frequency[expert_id] += 1

        # Update transitions (for prefetch)
        flat = routing_decisions.flatten()
        for i in range(len(flat) - self.top_k):
            expert = flat[i].item()
            next_layer_experts = flat[i + self.top_k:i + 2 * self.top_k].tolist()
            if expert not in self._transitions:
                self._transitions[expert] = Counter()
            for next_exp in next_layer_experts:
                self._transitions[expert][next_exp] += 1

        return fingerprint

    def _compute_fingerprint(self, routing_decisions: torch.Tensor) -> int:
        """Compute hash fingerprint for routing path."""
        # Flatten to [batch * seq, num_layers * top_k]
        flat = routing_decisions.flatten().cpu().numpy()
        # Simple hash from expert sequence
        fp = 0
        for i, exp in enumerate(flat):
            fp ^= hash((i, exp))
        return fp

    def predict_hot_experts(self, top_n: int = 8) -> list[int]:
        """Predict which experts to preload based on history."""
        return [e for e, _ in self._expert_frequency.most_common(top_n)]

    def get_next_experts(self, current_experts: list[int], top_n: int = 3) -> list[int]:
        """Get predicted next experts given current ones."""
        next_experts = Counter()
        for exp in current_experts:
            if exp in self._transitions:
                for next_exp, count in self._transitions[exp].most_common(top_n):
                    next_experts[next_exp] += count
        return [e for e, _ in next_experts.most_common(top_n)]

    def reset(self):
        """Reset history but keep learned patterns."""
        self._path_history.clear()

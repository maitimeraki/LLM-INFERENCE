import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


class RouterPredictor:
    """Predicts next expert activation using Markov chains and router bias."""

    def __init__(self, num_experts: int, num_top_predictions: int = 10):
        self.num_experts = num_experts
        self.num_top_predictions = num_top_predictions
        self.markov_matrix: Optional[np.ndarray] = None
        self.router_bias_history: List[np.ndarray] = []
        self.max_history = 100

    def build_markov_matrix(self, expert_sequences: List[List[int]]) -> None:
        """Build transition matrix from expert activation sequences."""
        transitions = {}
        for seq in expert_sequences:
            for i in range(len(seq) - 1):
                current = tuple(seq[i:i+1])
                next_expert = seq[i + 1]
                key = current
                if key not in transitions:
                    transitions[key] = np.zeros(self.num_experts)
                transitions[key][next_expert] += 1

        self.markov_matrix = np.zeros((self.num_experts, self.num_experts))
        for (current,), counts in transitions.items():
            self.markov_matrix[current] = counts / (counts.sum() + 1e-8)
        logger.info("Built Markov transition matrix")

    def record_router_bias(self, router_logits: np.ndarray) -> None:
        """Record router logits for bias tracking."""
        self.router_bias_history.append(router_logits)
        if len(self.router_bias_history) > self.max_history:
            self.router_bias_history.pop(0)

    def predict(self, current_expert_ids: List[int]) -> Tuple[List[int], List[float]]:
        """Predict next expert activations with confidence scores."""
        predictions = np.zeros(self.num_experts)

        # Markov prediction: if we have history
        if self.markov_matrix is not None and current_expert_ids:
            for eid in current_expert_ids:
                predictions += 0.4 * self.markov_matrix[eid]

        # Router bias trend: favor experts with rising logits
        if self.router_bias_history:
            recent_biases = np.stack(self.router_bias_history[-10:], axis=0)
            bias_trend = np.mean(recent_biases, axis=0)
            predictions += 0.3 * bias_trend / (bias_trend.max() + 1e-8)

        # Smoothing: uniform prior
        predictions += 0.3 / self.num_experts

        # Normalize to probabilities
        predictions = predictions / (predictions.sum() + 1e-8)

        # Return top-k
        top_indices = np.argsort(-predictions)[:self.num_top_predictions]
        top_scores = predictions[top_indices]
        return top_indices.tolist(), top_scores.tolist()

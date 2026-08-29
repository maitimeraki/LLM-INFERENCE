import logging
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class RouterPredictor:
    """Predict next experts using Markov chains and router bias history."""

    def __init__(self, num_experts: int = 256, markov_window: int = 100):
        self.num_experts = num_experts
        self.markov_window = markov_window

        # Markov transition matrix: (current_expert) -> {next_expert: probability}
        self.markov_matrix: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Router logit history per position
        self.router_bias_history: List[np.ndarray] = []

        # Co-activation frequency: (expert1, expert2) -> count
        self.co_activation_freq: Dict[Tuple[int, int], int] = defaultdict(int)

        # Expert sequence history for Markov learning
        self.expert_sequences: List[List[int]] = []

    def update_from_inference(self, activated_experts: List[int],
                              router_logits: Optional[np.ndarray] = None) -> None:
        """Update predictor with real inference data."""
        # Track expert sequences
        self.expert_sequences.append(activated_experts)
        if len(self.expert_sequences) > self.markov_window:
            self.expert_sequences.pop(0)

        # Update co-activation frequencies
        for i, exp1 in enumerate(activated_experts):
            for exp2 in activated_experts[i+1:]:
                key = tuple(sorted([exp1, exp2]))
                self.co_activation_freq[key] += 1

        # Update router logit history
        if router_logits is not None:
            self.router_bias_history.append(router_logits.copy())
            if len(self.router_bias_history) > self.markov_window:
                self.router_bias_history.pop(0)

        # Update Markov matrix
        self._update_markov_matrix()

    def _update_markov_matrix(self) -> None:
        """Build Markov transition matrix from expert sequences."""
        if len(self.expert_sequences) < 2:
            return

        for i in range(len(self.expert_sequences) - 1):
            current_experts = tuple(sorted(self.expert_sequences[i]))
            next_experts = self.expert_sequences[i + 1]

            for next_exp in next_experts:
                if current_experts not in self.markov_matrix:
                    self.markov_matrix[current_experts] = defaultdict(float)
                self.markov_matrix[current_experts][next_exp] += 1.0

        # Normalize to probabilities
        for current_state in self.markov_matrix:
            total = sum(self.markov_matrix[current_state].values())
            if total > 0:
                for next_exp in self.markov_matrix[current_state]:
                    self.markov_matrix[current_state][next_exp] /= total

    def predict_next_experts(self, current_experts: List[int],
                            top_k: int = 15,
                            confidence_threshold: float = 0.30) -> Tuple[List[int], List[float]]:
        """
        Predict top-k next experts with confidence scores.
        Combines Markov prediction, router bias, and co-activation frequency.
        """
        predictions: Dict[int, float] = defaultdict(float)

        # 1. Markov prediction (40% weight)
        current_state = tuple(sorted(current_experts))
        if current_state in self.markov_matrix:
            for next_exp, prob in self.markov_matrix[current_state].items():
                predictions[next_exp] += 0.40 * prob

        # 2. Co-activation frequency (35% weight)
        for current_exp in current_experts:
            for (exp1, exp2), count in self.co_activation_freq.items():
                if current_exp in (exp1, exp2):
                    other = exp1 if exp2 == current_exp else exp2
                    freq_score = min(count / max(1, len(self.expert_sequences)), 1.0)
                    predictions[other] += 0.35 * freq_score

        # 3. Router bias history (25% weight)
        if self.router_bias_history:
            recent_logits = np.mean([h for h in self.router_bias_history[-10:]], axis=0)
            max_logit = np.max(recent_logits)
            if max_logit > 0:
                for expert_id in range(min(len(recent_logits), self.num_experts)):
                    bias_score = recent_logits[expert_id] / max_logit
                    if bias_score > 0:
                        predictions[expert_id] += 0.25 * bias_score

        # Filter by confidence threshold and return top-k
        filtered = [(exp, score) for exp, score in predictions.items()
                   if score >= confidence_threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)

        top_experts = [exp for exp, _ in filtered[:top_k]]
        top_scores = [score for _, score in filtered[:top_k]]

        # Normalize scores to [0, 1]
        if top_scores:
            max_score = max(top_scores)
            top_scores = [s / max_score for s in top_scores]

        logger.debug(f"Predicted {len(top_experts)} experts: {top_experts[:5]}...")
        return top_experts, top_scores

    def get_co_activation_experts(self, expert_id: int,
                                  top_k: int = 8) -> Set[int]:
        """Get experts frequently co-activated with given expert."""
        co_experts: Dict[int, int] = defaultdict(int)
        for (exp1, exp2), count in self.co_activation_freq.items():
            if expert_id == exp1:
                co_experts[exp2] += count
            elif expert_id == exp2:
                co_experts[exp1] += count

        sorted_co = sorted(co_experts.items(), key=lambda x: x[1], reverse=True)
        return {exp for exp, _ in sorted_co[:top_k]}

    def stats(self) -> Dict:
        """Return predictor statistics."""
        return {
            "markov_states": len(self.markov_matrix),
            "co_activation_pairs": len(self.co_activation_freq),
            "history_size": len(self.expert_sequences),
            "router_logit_history": len(self.router_bias_history),
        }

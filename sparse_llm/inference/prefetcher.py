"""Predictive prefetcher for expert-centric MoE inference.

Predicts next experts based on routing patterns and prefetches them
in a background thread while compute runs, hiding I/O latency.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import Counter
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class PredictivePrefetcher:
    """
    Predict and prefetch experts based on routing patterns.

    Strategy:
    1. During prefill: learn which experts follow each other
    2. During decode: predict next experts from context
    3. Background thread: prefetch while compute runs
    """

    def __init__(
        self,
        expert_loader: Callable[[int, int], None],  # (expert_id, layer_id) -> None
        prefetch_queue_size: int = 16,
        enable_background_thread: bool = True,
        top_k_predictions: int = 3,
    ):
        """Initialize the prefetcher.

        Args:
            expert_loader: Callback to load expert weights (expert_id, layer_id)
            prefetch_queue_size: Max prefetch queue depth
            enable_background_thread: Whether to run background prefetch worker
            top_k_predictions: Number of top predicted experts to prefetch
        """
        self.expert_loader = expert_loader
        self.prefetch_queue: queue.Queue = queue.Queue(maxsize=prefetch_queue_size)
        self.transition_probs: dict[int, Counter] = {}
        self._stop_event = threading.Event()
        self.top_k_predictions = top_k_predictions

        if enable_background_thread:
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                daemon=True,
                name="PredictivePrefetcher"
            )
            self._prefetch_thread.start()
            logger.debug("PredictivePrefetcher background thread started")

    def observe(self, expert_id: int, next_expert_ids: list[int]) -> None:
        """Record transition probabilities from observed routing.

        Args:
            expert_id: Current expert that was activated
            next_expert_ids: Experts that followed in subsequent layers
        """
        if expert_id not in self.transition_probs:
            self.transition_probs[expert_id] = Counter()
        for next_id in next_expert_ids:
            self.transition_probs[expert_id][next_id] += 1

    def observe_layer(self, layer_id: int, expert_ids: list[int]) -> None:
        """Observe a full layer's expert activations.

        Args:
            layer_id: Layer index
            expert_ids: List of activated expert IDs for this layer
        """
        for i, exp in enumerate(expert_ids):
            next_experts = expert_ids[i+1:i+4] if i + 1 < len(expert_ids) else []
            self.observe(exp, next_experts)

    def predict_and_prefetch(self, current_experts: list[int], layer_id: int) -> None:
        """Predict next experts and queue them for prefetch.

        Args:
            current_experts: Currently activated expert IDs
            layer_id: Layer to prefetch for
        """
        for expert_id in current_experts:
            if expert_id in self.transition_probs:
                # Get top predicted next experts
                next_experts = [
                    exp for exp, _ in self.transition_probs[expert_id].most_common(self.top_k_predictions)
                ]
                for next_exp in next_experts:
                    try:
                        self.prefetch_queue.put_nowait((next_exp, layer_id))
                    except queue.Full:
                        # Queue full, skip remaining predictions
                        break

    def _prefetch_worker(self) -> None:
        """Background thread that prefetches queued experts."""
        while not self._stop_event.is_set():
            try:
                expert_id, layer_id = self.prefetch_queue.get(timeout=0.1)
                try:
                    self.expert_loader(expert_id, layer_id)
                except Exception as e:
                    logger.error(f"Prefetch error for expert {expert_id} layer {layer_id}: {e}")
            except queue.Empty:
                continue

    def stop(self) -> None:
        """Stop the background prefetch thread."""
        self._stop_event.set()
        if hasattr(self, "_prefetch_thread") and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
            logger.debug("PredictivePrefetcher background thread stopped")

    def get_stats(self) -> dict:
        """Return prefetcher statistics."""
        return {
            "num_transitions": len(self.transition_probs),
            "queue_size": self.prefetch_queue.qsize(),
            "queue_max_size": self.prefetch_queue.maxsize,
            "thread_alive": (
                hasattr(self, "_prefetch_thread")
                and self._prefetch_thread.is_alive()
            ),
        }

    def get_transition_probs(self) -> dict[int, list[tuple[int, int]]]:
        """Get transition probabilities for debugging/inspection.

        Returns:
            Dict mapping expert_id -> [(next_expert, count), ...]
        """
        return {
            exp: list(counter.most_common())
            for exp, counter in self.transition_probs.items()
        }


__all__ = ["PredictivePrefetcher"]

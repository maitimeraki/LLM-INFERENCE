"""MoE-specific cache manager with predictive prefetching.

This module wraps ExpertCache with MoE-aware logic including:
- Predictive prefetching using RouterPredictor
- Background expert preloading after prefill phase
- Cache optimization for decode phase
- Enhanced statistics for MoE workloads
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional, Set, Dict
import time

from sparse_llm.cache.expert_cache import ExpertCache, ExpertKey
try:
    from sparse_llm.core.router_predictor import RouterPredictor
except ImportError:
    from sparse_llm.core.router import RouterPredictor

logger = logging.getLogger(__name__)


class ExpertCacheManager:
    """Manages expert caching with predictive prefetching for MoE inference.

    Wraps ExpertCache with:
    1. MoE-specific preloading strategies
    2. Predictive prefetching using RouterPredictor
    3. Background expert loading during decode
    4. Aggregated statistics for cache performance
    """

    def __init__(
        self,
        expert_cache: ExpertCache,
        router_predictor: Optional[RouterPredictor] = None,
        enable_prefetch: bool = True,
        prefetch_confidence_threshold: float = 0.30,
        prefetch_top_k: int = 15,
    ):
        """Initialize the expert cache manager.

        Args:
            expert_cache: Underlying ExpertCache instance
            router_predictor: Optional RouterPredictor for predictive prefetching
            enable_prefetch: Whether to enable background prefetching
            prefetch_confidence_threshold: Minimum confidence for prefetching
            prefetch_top_k: Maximum number of experts to prefetch
        """
        self.cache = expert_cache
        self.router_predictor = router_predictor
        self.enable_prefetch = enable_prefetch
        self.prefetch_confidence_threshold = prefetch_confidence_threshold
        self.prefetch_top_k = prefetch_top_k

        # Prefetch state
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_queue: List[tuple[int, Optional[int], Callable]] = []
        self._prefetch_lock = threading.Lock()
        self._prefetch_shutdown = threading.Event()

        # Statistics
        self.prefetch_hits = 0  # Cache hits that were prefetched
        self.prefetch_misses = 0  # Prefetch predictions that weren't used
        self.last_prefetch_time = 0.0
        self.total_prefetch_time = 0.0

        # Track recent expert accesses for prediction
        self._recent_experts: List[int] = []
        self._recent_experts_lock = threading.Lock()

    def get(
        self,
        key_or_expert: ExpertKey | int,
        layer_id: Optional[int] = None,
    ) -> object | None:
        """Get an expert from cache.

        Args:
            key_or_expert: Expert ID or (layer_id, expert_id) tuple
            layer_id: Optional layer ID

        Returns:
            Cached expert weights or None if not found
        """
        result = self.cache.get(key_or_expert, layer_id)

        # Track access for prediction
        if result is not None:
            expert_id = key_or_expert if isinstance(key_or_expert, int) else key_or_expert[1]
            self._track_expert_access(expert_id)

        return result

    def put(
        self,
        key_or_expert: ExpertKey | int,
        weights: object,
        layer_id: Optional[int] = None,
        size_bytes: int | None = None,
        pin: bool = False,
    ) -> None:
        """Add an expert to cache.

        Args:
            key_or_expert: Expert ID or (layer_id, expert_id) tuple
            weights: Expert weights
            layer_id: Optional layer ID
            size_bytes: Optional size in bytes
            pin: Whether to pin the expert
        """
        self.cache.put(key_or_expert, weights, layer_id, size_bytes=size_bytes, _pin=pin)

    def get_or_load(
        self,
        key_or_expert: ExpertKey | int,
        loader: Callable[[], object],
        layer_id: Optional[int] = None,
        size_bytes: int | None = None,
        pin: bool = False,
    ) -> object:
        """Get expert from cache or load if missing.

        Args:
            key_or_expert: Expert ID or (layer_id, expert_id) tuple
            loader: Function to load expert if not cached
            layer_id: Optional layer ID
            size_bytes: Optional size in bytes
            pin: Whether to pin the expert

        Returns:
            Expert weights
        """
        result = self.cache.get_or_load(
            key_or_expert, loader, layer_id, size_bytes=size_bytes, pin=pin
        )

        # Track access for prediction
        expert_id = key_or_expert if isinstance(key_or_expert, int) else key_or_expert[1]
        self._track_expert_access(expert_id)

        return result

    def pin(self, key_or_expert: ExpertKey | int, layer_id: Optional[int] = None) -> None:
        """Pin an expert to prevent eviction."""
        self.cache.pin(key_or_expert, layer_id)

    def unpin(self, key_or_expert: ExpertKey | int, layer_id: Optional[int] = None) -> None:
        """Unpin an expert to allow eviction."""
        self.cache.unpin(key_or_expert, layer_id)

    def contains(self, key_or_expert: ExpertKey | int, layer_id: Optional[int] = None) -> bool:
        """Check if expert is cached."""
        return self.cache.contains(key_or_expert, layer_id)

    def preload_experts(
        self,
        expert_ids: List[int],
        loader: Callable[[int], object],
        layer_id: Optional[int] = None,
        pin: bool = False,
    ) -> None:
        """Preload a set of experts into cache.

        Args:
            expert_ids: List of expert IDs to preload
            loader: Function to load expert by ID
            layer_id: Optional layer ID
            pin: Whether to pin the preloaded experts
        """
        for expert_id in expert_ids:
            try:
                if not self.contains(expert_id, layer_id):
                    self.get_or_load(
                        expert_id,
                        lambda eid=expert_id: loader(eid),
                        layer_id=layer_id,
                        pin=pin,
                    )
                    logger.debug(f"Preloaded expert {expert_id} (layer {layer_id})")
            except Exception as e:
                logger.warning(f"Failed to preload expert {expert_id}: {e}")

    def optimize_cache_for_decode(
        self,
        current_experts: List[int],
        loader: Callable[[int], object],
        layer_id: Optional[int] = None,
    ) -> None:
        """Optimize cache for decode phase with predictive prefetching.

        After prefill, predict likely next experts and preload them in background.

        Args:
            current_experts: Currently active experts
            loader: Function to load expert by ID
            layer_id: Optional layer ID
        """
        if not self.enable_prefetch or self.router_predictor is None:
            return

        start_time = time.monotonic()

        # Predict next experts
        predicted_experts, confidence_scores = self.router_predictor.predict_next_experts(
            current_experts,
            top_k=self.prefetch_top_k,
            confidence_threshold=self.prefetch_confidence_threshold,
        )

        if not predicted_experts:
            logger.debug("No experts predicted for prefetching")
            return

        logger.info(
            f"Predicted {len(predicted_experts)} experts for prefetch: "
            f"{predicted_experts[:5]} (confidence: {confidence_scores[:5]})"
        )

        # Queue experts for background loading
        with self._prefetch_lock:
            for expert_id in predicted_experts:
                if not self.contains(expert_id, layer_id):
                    self._prefetch_queue.append((expert_id, layer_id, loader))

        # Start prefetch thread if not running
        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_worker,
                daemon=True,
                name="ExpertPrefetchWorker",
            )
            self._prefetch_thread.start()

        elapsed = time.monotonic() - start_time
        self.last_prefetch_time = elapsed
        self.total_prefetch_time += elapsed

    def _prefetch_worker(self) -> None:
        """Background worker that prefetches experts from queue."""
        logger.debug("Prefetch worker started")

        while not self._prefetch_shutdown.is_set():
            with self._prefetch_lock:
                if not self._prefetch_queue:
                    break
                expert_id, layer_id, loader = self._prefetch_queue.pop(0)

            try:
                # Check again in case it was loaded by another thread
                if not self.contains(expert_id, layer_id):
                    start = time.monotonic()
                    self.get_or_load(
                        expert_id,
                        lambda: loader(expert_id),
                        layer_id=layer_id,
                        pin=False,  # Don't pin prefetched experts
                    )
                    elapsed = time.monotonic() - start
                    logger.debug(
                        f"Prefetched expert {expert_id} (layer {layer_id}) in {elapsed:.3f}s"
                    )
            except Exception as e:
                logger.warning(f"Prefetch failed for expert {expert_id}: {e}")

        logger.debug("Prefetch worker finished")

    def update_predictor_from_inference(
        self,
        activated_experts: List[int],
        router_logits: Optional[object] = None,
    ) -> None:
        """Update router predictor with inference data.

        Args:
            activated_experts: List of expert IDs activated in this step
            router_logits: Optional router logits (numpy array)
        """
        if self.router_predictor is not None:
            self.router_predictor.update_from_inference(activated_experts, router_logits)

    def _track_expert_access(self, expert_id: int) -> None:
        """Track expert access for prediction."""
        with self._recent_experts_lock:
            self._recent_experts.append(expert_id)
            if len(self._recent_experts) > 100:
                self._recent_experts.pop(0)

    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache performance metrics including:
            - Basic cache stats (hit rate, occupancy, evictions)
            - Prefetch statistics (hits, misses, timing)
            - Router predictor statistics
        """
        base_stats = self.cache.stats()

        stats = {
            **base_stats,
            "prefetch_enabled": self.enable_prefetch,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "last_prefetch_time": self.last_prefetch_time,
            "total_prefetch_time": self.total_prefetch_time,
            "prefetch_queue_size": len(self._prefetch_queue),
        }

        if self.router_predictor is not None:
            stats["predictor"] = self.router_predictor.stats()

        return stats

    def clear(self) -> None:
        """Clear cache and reset statistics."""
        self.cache.clear()
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.last_prefetch_time = 0.0
        self.total_prefetch_time = 0.0
        with self._recent_experts_lock:
            self._recent_experts.clear()

    def shutdown(self) -> None:
        """Shutdown prefetch worker and cleanup resources."""
        self._prefetch_shutdown.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)


__all__ = ["ExpertCacheManager"]

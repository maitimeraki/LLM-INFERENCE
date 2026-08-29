import logging
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import heapq

logger = logging.getLogger(__name__)


class PrefetchLevel(Enum):
    L0 = 0  # Immediate, high priority
    L1 = 1  # Near-term, medium priority
    L2 = 2  # Speculative, low priority


@dataclass
class PrefetchTask:
    priority: int
    expert_id: int
    confidence: float
    level: PrefetchLevel

    def __lt__(self, other):
        """Comparison for priority queue (lower priority = higher urgency)."""
        return self.priority < other.priority


class PrefetchPipeline:
    """Orchestrate async expert prefetch across 3 priority levels."""

    def __init__(self, num_streams: int = 3):
        self.num_streams = num_streams
        self.l0_queue: List[PrefetchTask] = []
        self.l1_queue: List[PrefetchTask] = []
        self.l2_queue: List[PrefetchTask] = []

        self.pending_experts: Set[int] = set()
        self.completed_prefetches = 0
        self.cancelled_prefetches = 0

        logger.info(f"Initialized PrefetchPipeline with {num_streams} CUDA streams")

    def schedule_prefetch(self,
                         predicted_experts: List[int],
                         confidence_scores: List[float],
                         cached_experts: Set[int]) -> Dict[str, List[int]]:
        """
        Distribute experts across L0/L1/L2 based on confidence and cache status.
        Returns distribution: {L0: [exp_ids], L1: [exp_ids], L2: [exp_ids]}
        """
        distribution = {"L0": [], "L1": [], "L2": []}

        # Filter out already-cached experts
        to_prefetch = [(exp, conf) for exp, conf in zip(predicted_experts, confidence_scores)
                       if exp not in cached_experts]

        if not to_prefetch:
            logger.debug("All predicted experts already cached")
            return distribution

        # Distribute by confidence threshold
        for i, (expert_id, confidence) in enumerate(to_prefetch):
            if i < 3 and confidence > 0.85:
                distribution["L0"].append(expert_id)
                task = PrefetchTask(priority=0, expert_id=expert_id,
                                   confidence=confidence, level=PrefetchLevel.L0)
                heapq.heappush(self.l0_queue, task)
                self.pending_experts.add(expert_id)

            elif i < 11 and confidence > 0.60:
                distribution["L1"].append(expert_id)
                task = PrefetchTask(priority=1, expert_id=expert_id,
                                   confidence=confidence, level=PrefetchLevel.L1)
                heapq.heappush(self.l1_queue, task)
                self.pending_experts.add(expert_id)

            elif confidence > 0.30:
                distribution["L2"].append(expert_id)
                task = PrefetchTask(priority=2, expert_id=expert_id,
                                   confidence=confidence, level=PrefetchLevel.L2)
                heapq.heappush(self.l2_queue, task)
                self.pending_experts.add(expert_id)

        logger.debug(f"Scheduled prefetch: L0={len(distribution['L0'])}, "
                    f"L1={len(distribution['L1'])}, L2={len(distribution['L2'])}")
        return distribution

    def get_next_l0_task(self) -> Tuple[int, float]:
        """Get highest-priority L0 task."""
        if self.l0_queue:
            task = heapq.heappop(self.l0_queue)
            return task.expert_id, task.confidence
        return None, 0.0

    def get_next_l1_task(self) -> Tuple[int, float]:
        """Get highest-priority L1 task."""
        if self.l1_queue:
            task = heapq.heappop(self.l1_queue)
            return task.expert_id, task.confidence
        return None, 0.0

    def get_next_l2_task(self) -> Tuple[int, float]:
        """Get highest-priority L2 task (cancellable)."""
        if self.l2_queue:
            task = heapq.heappop(self.l2_queue)
            return task.expert_id, task.confidence
        return None, 0.0

    def mark_prefetch_complete(self, expert_id: int) -> None:
        """Mark expert as successfully prefetched."""
        if expert_id in self.pending_experts:
            self.pending_experts.remove(expert_id)
            self.completed_prefetches += 1
            logger.debug(f"Prefetch completed: expert {expert_id}")

    def mark_prefetch_cancelled(self, expert_id: int) -> None:
        """Mark expert as cancelled (memory pressure)."""
        if expert_id in self.pending_experts:
            self.pending_experts.remove(expert_id)
            self.cancelled_prefetches += 1
            logger.debug(f"Prefetch cancelled: expert {expert_id}")

    def clear(self) -> None:
        """Clear all pending prefetch tasks."""
        self.l0_queue.clear()
        self.l1_queue.clear()
        self.l2_queue.clear()
        self.pending_experts.clear()
        logger.info("Cleared prefetch pipeline")

    def stats(self) -> dict:
        """Return prefetch statistics."""
        return {
            "pending_experts": len(self.pending_experts),
            "l0_queue_size": len(self.l0_queue),
            "l1_queue_size": len(self.l1_queue),
            "l2_queue_size": len(self.l2_queue),
            "completed_prefetches": self.completed_prefetches,
            "cancelled_prefetches": self.cancelled_prefetches,
        }

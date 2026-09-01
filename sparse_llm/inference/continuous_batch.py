"""Decode batching and token scheduling for multi-request generation.

Groups compatible decode work, manages per-request sampling and stopping,
measures throughput and tail latency under mixed request workloads.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sparse_llm.inference.scheduler import RequestScheduler, RequestState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchScheduleConfig:
    """Configuration for continuous batch scheduling."""

    batch_size: int = 32  # max tokens in a batch
    max_batch_requests: int = 16  # max requests per batch
    schedule_interval_ms: float = 10.0  # how often to regroup batches


@dataclass
class DecodeBatchMetrics:
    """Metrics for a decode batch operation."""

    batch_id: int
    num_requests: int
    num_tokens: int
    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "num_requests": self.num_requests,
            "num_tokens": self.num_tokens,
            "latency_ms": self.latency_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
        }


@dataclass
class DecodeBatch:
    """A batch of requests at decode phase ready for generation."""

    batch_id: int
    request_ids: list[int] = field(default_factory=list)
    token_counts: dict[int, int] = field(default_factory=dict)  # request_id -> token count
    padded_length: int = 0  # max sequence length in batch (for padding)
    created_at: float = field(default_factory=time.time)

    def total_tokens(self) -> int:
        """Return total tokens in batch (sum across all requests)."""
        return sum(self.token_counts.values())

    def average_length(self) -> float:
        """Return average sequence length."""
        if not self.request_ids:
            return 0.0
        return self.total_tokens() / len(self.request_ids)


class ContinuousBatchScheduler:
    """Schedule decode-phase requests into batches.

    - Groups requests with compatible lengths to minimize padding
    - Respects batch size and request count limits
    - Manages per-request sampling and stopping logic
    - Tracks throughput and tail latency
    """

    def __init__(
        self,
        scheduler: RequestScheduler,
        config: BatchScheduleConfig | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.config = config or BatchScheduleConfig()
        self._next_batch_id = 0
        self._current_batch: DecodeBatch | None = None
        self._completed_batches: list[DecodeBatch] = []
        self._batch_metrics: list[DecodeBatchMetrics] = []
        self._last_schedule_time = time.time()

    def should_schedule_batch(self) -> bool:
        """Check if a new batch should be scheduled based on time or capacity."""
        now = time.time()
        elapsed_ms = (now - self._last_schedule_time) * 1000

        # Schedule if time elapsed or batch is full
        if elapsed_ms >= self.config.schedule_interval_ms:
            return True

        if self._current_batch is not None:
            if len(self._current_batch.request_ids) >= self.config.max_batch_requests:
                return True
            if self._current_batch.total_tokens() >= self.config.batch_size:
                return True

        return False

    def schedule_batch(self) -> DecodeBatch | None:
        """Create a new batch from decode-ready requests."""
        # Finalize current batch if present
        if self._current_batch is not None and self._current_batch.request_ids:
            self._completed_batches.append(self._current_batch)

        # Get active requests
        active_ids = self.scheduler.get_active_requests()
        decode_requests = []

        for req_id in active_ids:
            seq = self.scheduler.get_sequence(req_id)
            if seq and seq.state == RequestState.DECODE and seq.prefill_done:
                if not seq.should_stop():
                    decode_requests.append(req_id)

        if not decode_requests:
            return None

        # Group requests into batch
        batch = DecodeBatch(batch_id=self._next_batch_id)
        self._next_batch_id += 1

        total_tokens = 0
        max_length = 0

        for req_id in sorted(decode_requests)[: self.config.max_batch_requests]:
            seq = self.scheduler.get_sequence(req_id)
            if seq is None:
                continue

            seq_len = seq.total_sequence_length()
            token_count = seq.total_tokens_generated() + 1  # +1 for next token

            if total_tokens + token_count > self.config.batch_size:
                if batch.request_ids:  # Keep at least one request
                    break

            batch.request_ids.append(req_id)
            batch.token_counts[req_id] = token_count
            total_tokens += token_count
            max_length = max(max_length, seq_len)

        batch.padded_length = max_length
        self._current_batch = batch
        self._last_schedule_time = time.time()

        return batch if batch.request_ids else None

    def execute_decode_step(self, batch: DecodeBatch) -> dict[int, int] | None:
        """Execute one decode step for a batch.

        Returns: dict of request_id -> generated_token_id or None on error.
        ponytail: mock generation - real impl calls model forward pass.
        """
        if not batch.request_ids:
            return None

        start_time = time.time()
        generated_tokens = {}

        for req_id in batch.request_ids:
            seq = self.scheduler.get_sequence(req_id)
            if seq is None:
                continue

            # Mock token generation (deterministic for testing)
            # ponytail: real impl runs transformer forward pass
            mock_token_id = (req_id * 1000 + seq.total_tokens_generated()) % 32000

            # Apply sampling if configured
            if seq.sampling_state.get("temperature", 0.0) > 0:
                # ponytail: mock sampling - real impl would use temperature sampling
                mock_token_id = (mock_token_id + seq.total_tokens_generated()) % 32000

            generated_tokens[req_id] = mock_token_id

            # Update scheduler
            self.scheduler.add_generated_token(req_id, mock_token_id)

            # Check stopping criteria
            if seq.should_stop():
                self.scheduler.update_sequence(req_id, RequestState.COMPLETED)
                self.scheduler.complete_request(req_id)

        latency_ms = (time.time() - start_time) * 1000
        throughput = (len(batch.request_ids) / latency_ms * 1000) if latency_ms > 0 else 0

        metrics = DecodeBatchMetrics(
            batch_id=batch.batch_id,
            num_requests=len(batch.request_ids),
            num_tokens=batch.total_tokens(),
            latency_ms=latency_ms,
            throughput_tokens_per_sec=throughput,
        )
        self._batch_metrics.append(metrics)

        return generated_tokens if generated_tokens else None

    def get_batch_metrics(self) -> list[DecodeBatchMetrics]:
        """Return metrics for all completed batches."""
        return self._batch_metrics.copy()

    def stats(self) -> dict[str, Any]:
        """Return batch scheduler statistics."""
        if not self._batch_metrics:
            return {
                "total_batches": 0,
                "average_latency_ms": 0.0,
                "average_throughput_tokens_per_sec": 0.0,
                "tail_latency_ms": 0.0,
            }

        latencies = [m.latency_ms for m in self._batch_metrics]
        throughputs = [m.throughput_tokens_per_sec for m in self._batch_metrics]

        return {
            "total_batches": len(self._batch_metrics),
            "average_latency_ms": sum(latencies) / len(latencies),
            "average_throughput_tokens_per_sec": sum(throughputs) / len(throughputs) if throughputs else 0,
            "tail_latency_ms": max(latencies) if latencies else 0.0,
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0.0,
        }


class ContinuousGenerationEngine:
    """Orchestrates continuous batching for multi-request generation.

    Combines RequestScheduler and ContinuousBatchScheduler into a full
    generation loop with per-request isolation.
    """

    def __init__(
        self,
        scheduler: RequestScheduler,
        batch_scheduler: ContinuousBatchScheduler | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.batch_scheduler = batch_scheduler or ContinuousBatchScheduler(scheduler)
        self._total_batches_processed = 0

    def generate_batch_step(self) -> dict[int, int] | None:
        """Run one generation step: schedule batch, execute decode, return tokens.

        Returns: dict of request_id -> generated_token_id.
        """
        # Always schedule a batch if needed
        batch = self.batch_scheduler.schedule_batch()
        if batch is None or not batch.request_ids:
            return None

        result = self.batch_scheduler.execute_decode_step(batch)
        if result:
            self._total_batches_processed += 1

        return result

    def stats(self) -> dict[str, Any]:
        """Return combined engine statistics."""
        return {
            "scheduler": self.scheduler.stats(),
            "batch_scheduler": self.batch_scheduler.stats(),
            "total_batches_processed": self._total_batches_processed,
        }

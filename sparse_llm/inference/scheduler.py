"""Request scheduler with sequence state and paged KV cache management.

Handles request lifecycle: admission → generation → completion.
Tracks per-request KV cache, sampling state, and stopping criteria.
Manages KV page allocation and reclamation independently from expert cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class RequestState(Enum):
    """Lifecycle state of a request."""

    ADMITTED = "admitted"  # Request accepted, waiting for generation
    PREFILL = "prefill"  # Processing input tokens
    DECODE = "decode"  # Generating output tokens
    COMPLETED = "completed"  # Request finished


@dataclass(frozen=True)
class KVPageConfig:
    """Configuration for KV cache paging."""

    page_size: int = 128  # tokens per page
    max_pages_per_request: int = 16  # max KV pages per sequence
    max_total_pages: int = 512  # global KV page budget


@dataclass
class KVCacheState:
    """Per-request KV cache state."""

    request_id: int
    allocated_pages: list[int] = field(default_factory=list)  # page indices
    page_occupancy: int = 0  # tokens stored
    max_tokens: int = 0  # sequence length

    def num_pages_needed(self, total_tokens: int, page_size: int) -> int:
        """Calculate pages needed for total_tokens."""
        return (total_tokens + page_size - 1) // page_size

    def can_allocate(self, additional_tokens: int, page_size: int, max_pages: int) -> bool:
        """Check if additional tokens fit within page limits."""
        pages_needed = self.num_pages_needed(self.page_occupancy + additional_tokens, page_size)
        return pages_needed <= max_pages


@dataclass
class SequenceState:
    """Per-request generation state."""

    request_id: int
    prompt_tokens: torch.Tensor  # tokenized prompt [seq_len]
    generated_tokens: list[int] = field(default_factory=list)  # output token ids
    sampling_state: dict[str, Any] = field(default_factory=dict)  # temperature, top_k, etc.
    kv_cache: KVCacheState = field(default_factory=lambda: KVCacheState(request_id=0))
    stopping_criteria: dict[str, Any] = field(default_factory=dict)  # max_tokens, stop words
    state: RequestState = RequestState.ADMITTED
    prefill_done: bool = False

    def total_tokens_generated(self) -> int:
        """Return number of tokens generated so far."""
        return len(self.generated_tokens)

    def input_length(self) -> int:
        """Return length of input prompt."""
        return self.prompt_tokens.shape[0] if self.prompt_tokens.ndim > 0 else 0

    def total_sequence_length(self) -> int:
        """Return total sequence length (input + generated)."""
        return self.input_length() + self.total_tokens_generated()

    def should_stop(self) -> bool:
        """Check if generation should stop."""
        max_tokens = self.stopping_criteria.get("max_tokens", float("inf"))
        if self.total_tokens_generated() >= max_tokens:
            return True
        stop_words = self.stopping_criteria.get("stop_words", [])
        # ponytail: simplified - full impl would tokenize and check
        return False


@dataclass
class RequestSchedulerConfig:
    """Configuration for request scheduler."""

    max_requests: int = 32  # max concurrent requests
    kv_config: KVPageConfig = field(default_factory=KVPageConfig)


class RequestScheduler:
    """Manages request lifecycle, sequence state, and KV cache allocation.

    - Admits requests up to concurrency limit
    - Tracks per-request generation state
    - Allocates and reclaims KV pages
    - Isolates request state to prevent corruption
    """

    def __init__(self, config: RequestSchedulerConfig | None = None) -> None:
        self.config = config or RequestSchedulerConfig()
        self._next_request_id = 0
        self._requests: dict[int, SequenceState] = {}
        self._kv_page_allocator = KVPageAllocator(
            total_pages=self.config.kv_config.max_total_pages
        )
        self._metrics = {
            "total_admitted": 0,
            "total_completed": 0,
            "active_requests": 0,
        }

    def admit_request(
        self,
        prompt_tokens: torch.Tensor,
        sampling_state: dict[str, Any] | None = None,
        stopping_criteria: dict[str, Any] | None = None,
    ) -> int | None:
        """Admit a new request. Returns request_id or None if at capacity."""
        if len(self._requests) >= self.config.max_requests:
            return None

        request_id = self._next_request_id
        self._next_request_id += 1

        # Allocate KV pages for prefill + decode budget
        input_length = prompt_tokens.shape[0] if prompt_tokens.ndim > 0 else 0
        max_output_tokens = stopping_criteria.get("max_tokens", 128) if stopping_criteria else 128
        total_budget = input_length + max_output_tokens

        pages_needed = self.config.kv_config.page_size
        pages_needed = (total_budget + pages_needed - 1) // pages_needed
        pages_needed = min(pages_needed, self.config.kv_config.max_pages_per_request)

        allocated_pages = self._kv_page_allocator.allocate(pages_needed)
        if not allocated_pages:
            logger.warning(f"Failed to allocate KV pages for request {request_id}")
            return None

        kv_cache = KVCacheState(
            request_id=request_id,
            allocated_pages=allocated_pages,
            max_tokens=total_budget,
        )

        seq_state = SequenceState(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            sampling_state=sampling_state or {},
            stopping_criteria=stopping_criteria or {},
            kv_cache=kv_cache,
        )

        self._requests[request_id] = seq_state
        self._metrics["total_admitted"] += 1
        self._metrics["active_requests"] = len(self._requests)
        return request_id

    def get_sequence(self, request_id: int) -> SequenceState | None:
        """Retrieve sequence state for a request."""
        return self._requests.get(request_id)

    def update_sequence(self, request_id: int, new_state: RequestState) -> bool:
        """Update request state."""
        seq = self._requests.get(request_id)
        if seq is None:
            return False
        seq.state = new_state
        return True

    def add_generated_token(self, request_id: int, token_id: int) -> bool:
        """Add a generated token to a sequence."""
        seq = self._requests.get(request_id)
        if seq is None:
            return False
        seq.generated_tokens.append(token_id)
        return True

    def mark_prefill_done(self, request_id: int) -> bool:
        """Mark prefill phase complete for a request."""
        seq = self._requests.get(request_id)
        if seq is None:
            return False
        seq.prefill_done = True
        seq.state = RequestState.DECODE
        return True

    def complete_request(self, request_id: int) -> bool:
        """Mark request complete and reclaim KV pages."""
        seq = self._requests.get(request_id)
        if seq is None:
            return False

        # Reclaim KV pages
        self._kv_page_allocator.release(seq.kv_cache.allocated_pages)
        seq.state = RequestState.COMPLETED

        del self._requests[request_id]
        self._metrics["total_completed"] += 1
        self._metrics["active_requests"] = len(self._requests)
        return True

    def get_active_requests(self) -> list[int]:
        """Return list of active request IDs."""
        return list(self._requests.keys())

    def stats(self) -> dict[str, Any]:
        """Return scheduler metrics."""
        return self._metrics.copy()


class KVPageAllocator:
    """Simple allocator for KV cache pages with fragmentation tracking.

    ponytail: global pool, per-sequence tracking. Per-request allocation limits
    prevent one request from starving others.
    """

    def __init__(self, total_pages: int = 512) -> None:
        self.total_pages = total_pages
        self._free_pages = list(range(total_pages))
        self._allocated: dict[int, list[int]] = {}  # request_id -> page indices

    def allocate(self, num_pages: int) -> list[int] | None:
        """Allocate num_pages consecutive or scattered pages."""
        if num_pages > len(self._free_pages):
            return None

        allocated = self._free_pages[:num_pages]
        self._free_pages = self._free_pages[num_pages:]
        return allocated

    def release(self, pages: list[int]) -> None:
        """Release pages back to free pool."""
        self._free_pages.extend(pages)
        self._free_pages.sort()

    def free_pages_count(self) -> int:
        """Return number of free pages."""
        return len(self._free_pages)

    def utilization(self) -> float:
        """Return fraction of pages in use."""
        used = self.total_pages - len(self._free_pages)
        return used / self.total_pages if self.total_pages > 0 else 0.0

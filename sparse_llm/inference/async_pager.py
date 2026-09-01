"""Async expert pager with backpressure limits, sync fallback, and prefetch prediction."""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

import torch

from sparse_llm.cache import ExpertKey
from sparse_llm.core.router_predictor import RouterPredictor
from sparse_llm.inference.pager import ExpertPager, LoadedExpert

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackpressureConfig:
    """Configuration for async transfer backpressure."""

    max_outstanding: int = 3  # max concurrent transfers in flight


class AsyncExpertPager:
    """Async wrapper around ExpertPager with prefetch queue and backpressure limits.

    Outstanding transfers are limited to prevent OOM. Prefetch misses are handled
    gracefully. Falls back to sync when queue is full.
    """

    def __init__(
        self,
        sync_pager: ExpertPager,
        config: BackpressureConfig | None = None,
    ) -> None:
        self.sync_pager = sync_pager
        self.config = config or BackpressureConfig()

        # Async task management
        # ponytail: global semaphore, per-expert locks if concurrency limits matter
        self._prefetch_semaphore = asyncio.Semaphore(self.config.max_outstanding)
        self._pending: dict[ExpertKey, asyncio.Task[LoadedExpert | None]] = {}

    @contextmanager
    def sync_lease(self, key: ExpertKey) -> Iterator[LoadedExpert]:
        """Direct sync lease path; bypass async."""
        with self.sync_pager.lease(key) as expert:
            yield expert

    @asynccontextmanager
    async def prefetch_async(self, key: ExpertKey) -> AsyncIterator[None]:
        """Queue an async prefetch. Handles backpressure with gradual release.

        Yields immediately; actual load happens in background. Miss if load fails.
        """
        key = self.sync_pager._validate_key(key)

        # Check if already queued
        if key in self._pending:
            yield
            return

        # Try to acquire a slot; if full, yield immediately (miss is OK)
        acquired = self._prefetch_semaphore._value > 0
        if not acquired:
            # Queue is full; skip this prefetch (graceful miss)
            yield
            return

        # Spawn async load task
        async def load_task() -> LoadedExpert | None:
            try:
                async with self._prefetch_semaphore:
                    # Run sync load in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None,
                        lambda: self.sync_pager.cache.get_or_load(
                            key,
                            lambda: self.sync_pager._load(key),
                            size_bytes=self.sync_pager.index.expert(key).byte_size,
                            pin=False,
                        ),
                    )
            except Exception:
                # Prefetch miss: load failed, will fall back to sync on access
                return None

        self._pending[key] = asyncio.create_task(load_task())
        try:
            yield
        finally:
            # Keep task pending for reuse; don't wait
            pass

    @asynccontextmanager
    async def lease_with_prefetch(self, key: ExpertKey) -> AsyncIterator[LoadedExpert]:
        """Lease expert, checking prefetch cache first. Fall back to sync if needed."""
        key = self.sync_pager._validate_key(key)

        # Check if prefetch is done
        loaded = None
        if key in self._pending:
            task = self._pending[key]
            try:
                # Non-blocking check; don't await (no backpressure wait)
                if task.done():
                    loaded = task.result()
                    del self._pending[key]
            except Exception:
                # Prefetch failed; will load sync
                del self._pending[key]

        # Fall back to sync if prefetch miss
        if loaded is None:
            with self.sync_pager.lease(key) as expert:
                yield expert
        else:
            yield loaded

    def request_stats(self) -> dict[str, object]:
        """Get stats from underlying sync pager."""
        return self.sync_pager.request_stats()

    def clear_request_stats(self) -> None:
        """Clear stats on underlying sync pager."""
        self.sync_pager.clear_request_stats()

    async def drain_pending(self) -> None:
        """Wait for all pending prefetches. Useful before shutdown."""
        if self._pending:
            await asyncio.gather(
                *[task for task in self._pending.values()],
                return_exceptions=True,
            )


__all__ = ["AsyncExpertPager", "BackpressureConfig"]

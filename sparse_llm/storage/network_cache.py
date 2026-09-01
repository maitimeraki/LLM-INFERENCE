"""Network-backed storage with immutable expert identities and checksum validation.

Provides a caching layer that validates checksums, handles local-disk cache
fallback to remote fetch, and maintains immutable expert page identities.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class ExpertPageIdentity:
    """Immutable identity for an expert page stored remotely or locally.

    Identifies a unique expert version across checkpoint versions and formats.
    """

    checkpoint_version: str
    layer_id: int
    expert_id: int
    quantization_format: str = "float32"
    schema_version: int = 1

    @property
    def unique_id(self) -> str:
        """Generate a unique identifier for this expert page."""
        parts = [
            self.checkpoint_version,
            f"layer_{self.layer_id:04d}",
            f"expert_{self.expert_id:04d}",
            self.quantization_format,
        ]
        return "_".join(parts)


@dataclass(frozen=True)
class ExpertPageRecord:
    """Record of an expert page with metadata and checksum."""

    identity: ExpertPageIdentity
    byte_size: int
    checksum_sha256: Optional[str] = None

    def validate_checksum(self, data: bytes) -> bool:
        """Validate data against stored checksum.

        Args:
            data: Raw bytes of the expert page.

        Returns:
            True if checksum matches or no checksum is stored, False otherwise.
        """
        if self.checksum_sha256 is None:
            return True
        computed = hashlib.sha256(data).hexdigest()
        return computed == self.checksum_sha256

    def compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()


class NetworkCache:
    """Local cache with network fallback for expert pages.

    Manages local disk cache, validates checksums, and provides fallback
    to remote storage when local cache misses.
    """

    def __init__(
        self,
        local_cache_dir: str | Path = "./cache/network_experts",
        remote_fetch_fn: Optional[callable] = None,
        enable_checksum: bool = True,
    ):
        """Initialize network cache.

        Args:
            local_cache_dir: Local directory for cached experts.
            remote_fetch_fn: Callable(identity: ExpertPageIdentity) -> bytes.
              If None, network fallback is disabled.
            enable_checksum: Whether to validate checksums.
        """
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.remote_fetch_fn = remote_fetch_fn
        self.enable_checksum = enable_checksum
        self.page_records: dict[str, ExpertPageRecord] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "remote_fetches": 0, "checksum_failures": 0}

    def register_page(
        self,
        identity: ExpertPageIdentity,
        byte_size: int,
        checksum: Optional[str] = None,
    ) -> None:
        """Register an expert page with metadata.

        Args:
            identity: ExpertPageIdentity for this page.
            byte_size: Expected size in bytes.
            checksum: Optional SHA256 checksum.
        """
        record = ExpertPageRecord(
            identity=identity, byte_size=byte_size, checksum_sha256=checksum
        )
        self.page_records[identity.unique_id] = record

    def get_page_path(self, identity: ExpertPageIdentity) -> Path:
        """Resolve local cache path for an expert page."""
        return self.local_cache_dir / f"{identity.unique_id}.pt"

    def load_expert(self, identity: ExpertPageIdentity) -> bytes:
        """Load expert page from cache or remote.

        Args:
            identity: ExpertPageIdentity to load.

        Returns:
            Raw bytes of the expert page.

        Raises:
            KeyError: If page not registered.
            ValueError: If checksum validation fails.
            RuntimeError: If page not in cache and remote fetch unavailable.
        """
        record = self.page_records[identity.unique_id]
        cache_path = self.get_page_path(identity)

        # Try local cache first.
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                data = f.read()
            if self.enable_checksum and not record.validate_checksum(data):
                self.cache_stats["checksum_failures"] += 1
                logger.warning(f"Checksum validation failed for {identity.unique_id}")
                raise ValueError(f"Checksum mismatch for {identity.unique_id}")
            self.cache_stats["hits"] += 1
            return data

        # Fall back to remote fetch.
        self.cache_stats["misses"] += 1
        if self.remote_fetch_fn is None:
            raise RuntimeError(
                f"Expert {identity.unique_id} not in local cache and remote fetch unavailable"
            )

        logger.info(f"Fetching {identity.unique_id} from remote storage")
        data = self.remote_fetch_fn(identity)
        self.cache_stats["remote_fetches"] += 1

        if self.enable_checksum and not record.validate_checksum(data):
            self.cache_stats["checksum_failures"] += 1
            logger.warning(f"Checksum validation failed for remote {identity.unique_id}")
            raise ValueError(f"Checksum mismatch for remote {identity.unique_id}")

        # Persist to local cache for future use.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)
        logger.debug(f"Cached {identity.unique_id} locally")

        return data

    def save_expert(self, identity: ExpertPageIdentity, data: bytes) -> None:
        """Save expert page to local cache.

        Args:
            identity: ExpertPageIdentity for this page.
            data: Raw bytes to cache.
        """
        self.register_page(
            identity,
            byte_size=len(data),
            checksum=hashlib.sha256(data).hexdigest(),
        )
        cache_path = self.get_page_path(identity)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(data)

    def get_stats(self) -> dict[str, object]:
        """Return cache statistics."""
        return {
            **self.cache_stats,
            "local_cache_dir": str(self.local_cache_dir),
            "registered_pages": len(self.page_records),
        }

    def clear_stats(self) -> None:
        """Reset cache statistics."""
        self.cache_stats = {"hits": 0, "misses": 0, "remote_fetches": 0, "checksum_failures": 0}

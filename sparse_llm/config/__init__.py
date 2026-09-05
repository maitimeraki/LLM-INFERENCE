"""Configuration for the model-independent inference baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceConfig:
    """Explicit options accepted by the canonical inference runtime."""

    model: str
    device: str = "auto"
    dtype: str | None = None
    revision: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    device_map: str | None = None
    offload_folder: str | None = None
    expert_cache_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return configuration suitable for logs or JSON diagnostics."""
        return asdict(self)


__all__ = ["InferenceConfig"]

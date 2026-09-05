"""Resource budget dataclasses for hardware capacity tracking."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GPUInfo:
    """Information about a single GPU device."""
    device_id: int
    total_bytes: int
    available_bytes: int
    compute_capability: tuple[int, int]
    name: str


@dataclass(frozen=True)
class CPUInfo:
    """Information about CPU memory."""
    total_bytes: int
    available_bytes: int
    usable_bytes: int  # After safety margin


@dataclass(frozen=True)
class StorageInfo:
    """Information about storage location for cold experts."""
    path: str
    available_bytes: int
    is_ssd: bool
    estimated_bandwidth_mbps: int


@dataclass
class ResourceBudget:
    """Complete hardware resource budget for weight placement."""
    gpus: list[GPUInfo]
    cpu: CPUInfo
    storage: StorageInfo
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Generate warnings
        warnings = []

        # CPU RAM check (minimum 8GB recommended)
        if self.cpu.usable_bytes < 8 * 1024**3:
            warnings.append(
                f"CPU RAM below recommended minimum (8GB). "
                f"Available: {self.cpu.usable_bytes / 1024**3:.1f}GB. "
                f"May experience poor performance or OOM."
            )

        # Storage check (minimum 10GB)
        if self.storage.available_bytes < 10 * 1024**3:
            warnings.append(
                f"Storage space below minimum (10GB). "
                f"Available: {self.storage.available_bytes / 1024**3:.1f}GB. "
                f"Cannot store cold experts on disk."
            )

        # HDD warning
        if not self.storage.is_ssd:
            warnings.append(
                "Storage is HDD (not SSD). Expert loading will be 5-10x slower. "
                "Consider moving model cache to SSD for better performance."
            )

        object.__setattr__(self, 'warnings', warnings)

    @property
    def total_gpu_bytes(self) -> int:
        """Total available GPU memory across all devices (with safety margin already applied)."""
        return sum(gpu.available_bytes for gpu in self.gpus)

    @property
    def total_cpu_bytes(self) -> int:
        """Total usable CPU memory (with safety margin already applied)."""
        return self.cpu.usable_bytes

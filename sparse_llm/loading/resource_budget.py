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

    @classmethod
    def from_system(cls) -> ResourceBudget:
        """Detect and return system hardware resources."""
        import torch
        import psutil
        import os
        from pathlib import Path

        # Detect GPUs
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory
                # Reserve 10% for CUDA overhead
                available = int(total * 0.9)
                gpus.append(GPUInfo(
                    device_id=i,
                    total_bytes=total,
                    available_bytes=available,
                    compute_capability=(props.major, props.minor),
                    name=props.name
                ))

        # Detect CPU memory
        mem = psutil.virtual_memory()
        # Reserve 20% safety margin for OS and other processes
        usable = int(mem.available * 0.8)
        cpu_info = CPUInfo(
            total_bytes=mem.total,
            available_bytes=mem.available,
            usable_bytes=usable
        )

        # Detect storage (use temp dir or current dir)
        cache_dir = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))
        disk = psutil.disk_usage(str(cache_dir) if cache_dir.exists() else str(Path.cwd()))

        # Try to detect if SSD (heuristic: check if path contains 'ssd' or default to True for modern systems)
        is_ssd = True  # Assume SSD by default on modern systems
        try:
            # On Windows, check disk type
            if os.name == 'nt':
                import subprocess
                result = subprocess.run(['powershell', '-Command',
                    f'Get-PhysicalDisk | Where-Object {{$_.DeviceId -eq 0}} | Select-Object -ExpandProperty MediaType'],
                    capture_output=True, text=True, timeout=2)
                if 'SSD' in result.stdout or 'NVMe' in result.stdout:
                    is_ssd = True
                elif 'HDD' in result.stdout:
                    is_ssd = False
        except:
            pass  # Keep default assumption

        storage_info = StorageInfo(
            path=str(cache_dir),
            available_bytes=disk.free,
            is_ssd=is_ssd,
            estimated_bandwidth_mbps=5000 if is_ssd else 150  # Typical SSD vs HDD speeds
        )

        return cls(gpus=gpus, cpu=cpu_info, storage=storage_info)

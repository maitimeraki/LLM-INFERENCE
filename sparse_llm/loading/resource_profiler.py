"""System resource profiling with hardware detection and safety margins."""

from __future__ import annotations
import platform
from pathlib import Path

import psutil
import torch

from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo


class ResourceProfiler:
    """Detect and measure all compute resources on the system."""

    def __init__(self, *, gpu_margin: float = 0.85, cpu_margin: float = 0.80, storage_margin: float = 0.90):
        """Initialize profiler with safety margins.

        Args:
            gpu_margin: Use this fraction of available GPU memory (default 85%)
            cpu_margin: Use this fraction of available CPU memory (default 80%)
            storage_margin: Use this fraction of available storage (default 90%)
        """
        self.gpu_margin = gpu_margin
        self.cpu_margin = cpu_margin
        self.storage_margin = storage_margin

    def profile(self, storage_path: str | None = None) -> ResourceBudget:
        """Profile system resources and return budget.

        Args:
            storage_path: Optional custom path for expert cache (default: ~/.cache/sparse_llm/experts)

        Returns:
            ResourceBudget with detected hardware and applied safety margins
        """
        gpus = self._detect_gpus()
        cpu = self._detect_cpu()
        storage = self._detect_storage(storage_path)

        return ResourceBudget(gpus=gpus, cpu=cpu, storage=storage)

    def _detect_gpus(self) -> list[GPUInfo]:
        """Detect all CUDA GPUs and measure available memory."""
        gpus = []

        if not torch.cuda.is_available():
            return gpus

        for device_id in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(device_id)
            total_bytes = props.total_memory

            # Get current free memory
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            free_bytes = torch.cuda.mem_get_info()[0]

            # Apply safety margin
            available_bytes = int(free_bytes * self.gpu_margin)

            gpu = GPUInfo(
                device_id=device_id,
                total_bytes=total_bytes,
                available_bytes=available_bytes,
                compute_capability=(props.major, props.minor),
                name=props.name
            )
            gpus.append(gpu)

        return gpus

    def _detect_cpu(self) -> CPUInfo:
        """Detect CPU memory and apply safety margin."""
        mem = psutil.virtual_memory()
        total_bytes = mem.total
        available_bytes = mem.available

        # Apply safety margin (use 80% of available to leave room for OS and other processes)
        usable_bytes = int(available_bytes * self.cpu_margin)

        return CPUInfo(
            total_bytes=total_bytes,
            available_bytes=available_bytes,
            usable_bytes=usable_bytes
        )

    def _detect_storage(self, storage_path: str | None) -> StorageInfo:
        """Detect storage location and measure available space."""
        # Default storage path
        if storage_path is None:
            storage_path = str(Path.home() / ".cache" / "sparse_llm" / "experts")

        # Create path if it doesn't exist
        Path(storage_path).mkdir(parents=True, exist_ok=True)

        # Get disk usage
        usage = psutil.disk_usage(storage_path)
        available_bytes = int(usage.free * self.storage_margin)

        # Detect SSD vs HDD
        is_ssd = self._is_ssd(storage_path)

        # Estimate bandwidth
        estimated_bandwidth_mbps = 1200 if is_ssd else 120  # SSD: 1200MB/s, HDD: 120MB/s

        return StorageInfo(
            path=storage_path,
            available_bytes=available_bytes,
            is_ssd=is_ssd,
            estimated_bandwidth_mbps=estimated_bandwidth_mbps
        )

    def _is_ssd(self, path: str) -> bool:
        """Detect if path is on SSD or HDD."""
        # Platform-specific detection
        system = platform.system()

        if system == "Linux":
            return self._is_ssd_linux(path)
        elif system == "Windows":
            return self._is_ssd_windows(path)
        else:
            # macOS or unknown: assume SSD (optimistic)
            return True

    def _is_ssd_linux(self, path: str) -> bool:
        """Linux SSD detection via /sys/block."""
        try:
            # Get device for path
            import subprocess
            result = subprocess.run(
                ["df", path],
                capture_output=True,
                text=True,
                check=True
            )
            device_line = result.stdout.split('\n')[1]
            device = device_line.split()[0].replace("/dev/", "")

            # Remove partition number
            device_base = device.rstrip('0123456789')

            # Check rotational flag
            rotational_path = f"/sys/block/{device_base}/queue/rotational"
            if Path(rotational_path).exists():
                with open(rotational_path) as f:
                    return f.read().strip() == "0"
        except Exception:
            pass

        # Fallback: assume SSD
        return True

    def _is_ssd_windows(self, path: str) -> bool:
        """Windows SSD detection via PowerShell."""
        try:
            import subprocess
            # Get drive letter
            drive = Path(path).resolve().drive

            # Query via PowerShell
            cmd = f'powershell "Get-PhysicalDisk | Where-Object {{$_.DeviceId -eq (Get-Partition -DriveLetter {drive[0]}).DiskNumber}} | Select-Object -ExpandProperty MediaType"'
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True, check=True)
            media_type = result.stdout.strip()

            return "SSD" in media_type
        except Exception:
            pass

        # Fallback: assume SSD
        return True

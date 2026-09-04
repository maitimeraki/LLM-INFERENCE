"""Configuration management for SparseLLM server.

Loads configuration from YAML file or environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TieringConfig:
    """Expert tiering configuration."""

    gpu_cache_gb: float = 2.0
    cpu_cache_gb: float = 14.0
    disk_cache_path: str = "./expert_cache"
    prefetch_workers: int = 2


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"


@dataclass
class OptimizationConfig:
    """Optimization configuration."""

    prefetch_enabled: bool = True
    batch_size: int = 4
    max_concurrent_requests: int = 10


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""

    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    log_format: str = "json"  # "json" or "text"
    log_file: Optional[str] = None
    metrics_retention_samples: int = 1000


@dataclass
class ResourceAwareConfig:
    """Resource-aware loading configuration."""

    enabled: bool = False  # Default off for backward compatibility
    storage_path: Optional[str] = None  # None = use default ~/.cache/sparse_llm/experts
    gpu_margin: float = 0.85  # Use 85% of available GPU
    cpu_margin: float = 0.80  # Use 80% of available CPU
    storage_margin: float = 0.90  # Use 90% of available storage


@dataclass
class SparseLLMConfig:
    """Main configuration for SparseLLM server."""

    model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    device: str = "auto"
    dtype: Optional[str] = None

    tiering: TieringConfig = field(default_factory=TieringConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    resource_aware: ResourceAwareConfig = field(default_factory=ResourceAwareConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SparseLLMConfig:
        """Load configuration from YAML file."""

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> SparseLLMConfig:
        """Load configuration from dictionary."""

        return cls(
            model=data.get("model", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
            device=data.get("device", "auto"),
            dtype=data.get("dtype"),
            tiering=TieringConfig(**data.get("tiering", {})),
            server=ServerConfig(**data.get("server", {})),
            optimization=OptimizationConfig(**data.get("optimization", {})),
            monitoring=MonitoringConfig(**data.get("monitoring", {})),
            resource_aware=ResourceAwareConfig(**data.get("resource_aware", {})),
        )

    @classmethod
    def from_env(cls) -> SparseLLMConfig:
        """Load configuration from environment variables."""

        return cls(
            model=os.getenv("MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
            device=os.getenv("DEVICE", "auto"),
            dtype=os.getenv("DTYPE"),
            tiering=TieringConfig(
                gpu_cache_gb=float(os.getenv("GPU_CACHE_GB", "2.0")),
                cpu_cache_gb=float(os.getenv("CPU_CACHE_GB", "14.0")),
                disk_cache_path=os.getenv("DISK_CACHE_PATH", "./expert_cache"),
                prefetch_workers=int(os.getenv("PREFETCH_WORKERS", "2")),
            ),
            server=ServerConfig(
                host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8000")),
                workers=int(os.getenv("WORKERS", "1")),
                log_level=os.getenv("LOG_LEVEL", "info"),
            ),
            optimization=OptimizationConfig(
                prefetch_enabled=os.getenv("PREFETCH_ENABLED", "true").lower() == "true",
                batch_size=int(os.getenv("BATCH_SIZE", "4")),
                max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
            ),
            monitoring=MonitoringConfig(
                prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
                prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
                log_format=os.getenv("LOG_FORMAT", "json"),
                log_file=os.getenv("LOG_FILE"),
                metrics_retention_samples=int(os.getenv("METRICS_RETENTION", "1000")),
            ),
            resource_aware=ResourceAwareConfig(
                enabled=os.getenv("RESOURCE_AWARE_ENABLED", "false").lower() == "true",
                storage_path=os.getenv("RESOURCE_AWARE_STORAGE_PATH"),
                gpu_margin=float(os.getenv("RESOURCE_AWARE_GPU_MARGIN", "0.85")),
                cpu_margin=float(os.getenv("RESOURCE_AWARE_CPU_MARGIN", "0.80")),
                storage_margin=float(os.getenv("RESOURCE_AWARE_STORAGE_MARGIN", "0.90")),
            ),
        )

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> SparseLLMConfig:
        """Load configuration with fallback priority:

        1. YAML file if provided
        2. ./config.yaml if exists
        3. Environment variables
        4. Defaults
        """

        # Try explicit config path
        if config_path:
            return cls.from_yaml(config_path)

        # Try default config.yaml
        default_config = Path("config.yaml")
        if default_config.exists():
            return cls.from_yaml(default_config)

        # Fall back to environment variables
        return cls.from_env()

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""

        return {
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "tiering": {
                "gpu_cache_gb": self.tiering.gpu_cache_gb,
                "cpu_cache_gb": self.tiering.cpu_cache_gb,
                "disk_cache_path": self.tiering.disk_cache_path,
                "prefetch_workers": self.tiering.prefetch_workers,
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "log_level": self.server.log_level,
            },
            "optimization": {
                "prefetch_enabled": self.optimization.prefetch_enabled,
                "batch_size": self.optimization.batch_size,
                "max_concurrent_requests": self.optimization.max_concurrent_requests,
            },
            "monitoring": {
                "prometheus_enabled": self.monitoring.prometheus_enabled,
                "prometheus_port": self.monitoring.prometheus_port,
                "log_format": self.monitoring.log_format,
                "log_file": self.monitoring.log_file,
                "metrics_retention_samples": self.monitoring.metrics_retention_samples,
            },
            "resource_aware": {
                "enabled": self.resource_aware.enabled,
                "storage_path": self.resource_aware.storage_path,
                "gpu_margin": self.resource_aware.gpu_margin,
                "cpu_margin": self.resource_aware.cpu_margin,
                "storage_margin": self.resource_aware.storage_margin,
            },
        }

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""

        errors = []

        if self.tiering.gpu_cache_gb < 0:
            errors.append("gpu_cache_gb must be non-negative")

        if self.tiering.cpu_cache_gb < 0:
            errors.append("cpu_cache_gb must be non-negative")

        if self.server.port < 1 or self.server.port > 65535:
            errors.append("port must be between 1 and 65535")

        if self.server.workers < 1:
            errors.append("workers must be at least 1")

        if self.optimization.batch_size < 1:
            errors.append("batch_size must be at least 1")

        if self.optimization.max_concurrent_requests < 1:
            errors.append("max_concurrent_requests must be at least 1")

        return errors


def create_default_config(output_path: str = "config.yaml") -> None:
    """Create a default configuration file."""

    config = SparseLLMConfig()

    config_dict = config.to_dict()

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Default configuration created at: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create":
        output = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        create_default_config(output)
    else:
        # Load and validate config
        config = SparseLLMConfig.load()
        errors = config.validate()

        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        else:
            print("Configuration valid:")
            print(yaml.dump(config.to_dict(), default_flow_style=False))

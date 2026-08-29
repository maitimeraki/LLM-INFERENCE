from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class QuantizationBits(Enum):
    FP32 = 32
    FP16 = 16
    INT8 = 8
    INT4 = 4


class StorageBackend(Enum):
    LOCAL_SSD = "local_ssd"
    NETWORK = "network"
    S3 = "s3"


@dataclass
class CacheConfig:
    expert_count_target: int = 22
    quantization_bits: int = 4
    recency_weight: float = 0.30
    frequency_weight: float = 0.35
    predictability_weight: float = 0.20
    batch_load_weight: float = 0.15
    zombie_timeout_ms: int = 100


@dataclass
class PrefetchConfig:
    l0_count: int = 3
    l0_latency_ms: int = 5
    l1_count: int = 8
    l1_latency_ms: int = 50
    l2_count: int = 10
    l2_latency_ms: int = 500
    cuda_stream_count: int = 3


@dataclass
class StorageConfig:
    backend: StorageBackend = StorageBackend.LOCAL_SSD
    primary_path: str = "./cache/experts"
    fallback_path: str = "./cache/swap"
    network_endpoint: Optional[str] = None


@dataclass
class ResourceConfig:
    vram_target_gb: int = 6
    gpu_warning_percent: float = 75.0
    gpu_critical_percent: float = 90.0
    cpu_warning_percent: float = 70.0
    cpu_critical_percent: float = 85.0


@dataclass
class SchedulingConfig:
    max_batch_size: int = 16
    max_queue_size: int = 1000
    request_timeout_sec: float = 300.0
    prefetch_concurrency: int = 4


@dataclass
class InferenceConfig:
    model_name: str = "meta-llama/Llama-2-100b-moe"
    model_dtype: str = "float16"
    cache: CacheConfig = field(default_factory=CacheConfig)
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)

    enable_prediction: bool = True
    enable_prefetch: bool = True
    enable_quantization: bool = True
    enable_production: bool = True
    enable_monitoring: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_dtype": self.model_dtype,
            "cache": self.cache.__dict__,
            "prefetch": self.prefetch.__dict__,
            "storage": self.storage.__dict__,
            "resource": self.resource.__dict__,
            "scheduling": self.scheduling.__dict__,
            "features": {
                "prediction": self.enable_prediction,
                "prefetch": self.enable_prefetch,
                "quantization": self.enable_quantization,
                "production": self.enable_production,
                "monitoring": self.enable_monitoring,
            },
        }


__all__ = [
    "InferenceConfig",
    "CacheConfig",
    "PrefetchConfig",
    "StorageConfig",
    "ResourceConfig",
    "SchedulingConfig",
    "QuantizationBits",
    "StorageBackend",
]

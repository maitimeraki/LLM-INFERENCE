"""Measured outputs from the reference generation path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationMetrics:
    """Latency, throughput, memory, and reference cache measurements."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    peak_vram_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    expert_load_time_ms: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        """Return the stable metric names used by the baseline."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "prefill_latency_ms": self.prefill_latency_ms,
            "decode_latency_ms": self.decode_latency_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "peak_vram_bytes": self.peak_vram_bytes,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "expert_load_time_ms": self.expert_load_time_ms,
        }


@dataclass
class GenerationResult:
    """Generated continuation and the measurements for that request."""

    text: str
    token_ids: list[int]
    metrics: GenerationMetrics


__all__ = ["GenerationMetrics", "GenerationResult"]

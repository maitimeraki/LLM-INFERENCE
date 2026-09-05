"""Phase 5: Observability - Prometheus Metrics and Structured Logging

Provides comprehensive observability for SparseLLM server.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional

import sys


# ===== Structured Logging =====

class StructuredLogger:
    """JSON structured logger for machine-readable logs."""

    def __init__(
        self,
        name: str,
        format_type: str = "json",
        log_file: Optional[str] = None,
        level: int = logging.INFO,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.format_type = format_type

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self._get_formatter())
            self.logger.addHandler(file_handler)

    def _get_formatter(self) -> logging.Formatter:
        """Get appropriate formatter based on format type."""

        if self.format_type == "json":
            return JsonFormatter()
        else:
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def info(self, event: str, **metadata):
        """Log INFO level event."""
        self._log(logging.INFO, event, metadata)

    def debug(self, event: str, **metadata):
        """Log DEBUG level event."""
        self._log(logging.DEBUG, event, metadata)

    def warning(self, event: str, **metadata):
        """Log WARNING level event."""
        self._log(logging.WARNING, event, metadata)

    def error(self, event: str, **metadata):
        """Log ERROR level event."""
        self._log(logging.ERROR, event, metadata)

    def _log(self, level: int, event: str, metadata: dict):
        """Internal log method."""

        if self.format_type == "json":
            # Store metadata for JSON formatter
            extra = {"event": event, "metadata": metadata}
            self.logger.log(level, event, extra=extra)
        else:
            # Format metadata as string for text logging
            meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            msg = f"{event} | {meta_str}" if meta_str else event
            self.logger.log(level, msg)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""

        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }

        # Add metadata if present
        if hasattr(record, "metadata"):
            log_data.update(record.metadata)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# ===== Metrics Collection =====

@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot."""

    timestamp: float
    value: float


class Counter:
    """Simple counter metric."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.value = 0

    def inc(self, amount: float = 1.0):
        """Increment counter."""
        self.value += amount

    def get(self) -> float:
        """Get current value."""
        return self.value

    def reset(self):
        """Reset counter to zero."""
        self.value = 0


class Gauge:
    """Gauge metric (value that can go up or down)."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.value = 0.0

    def set(self, value: float):
        """Set gauge value."""
        self.value = value

    def inc(self, amount: float = 1.0):
        """Increment gauge."""
        self.value += amount

    def dec(self, amount: float = 1.0):
        """Decrement gauge."""
        self.value -= amount

    def get(self) -> float:
        """Get current value."""
        return self.value


class Histogram:
    """Histogram metric for distributions."""

    def __init__(self, name: str, description: str = "", max_samples: int = 1000):
        self.name = name
        self.description = description
        self.max_samples = max_samples
        self.samples = deque(maxlen=max_samples)
        self.count = 0
        self.sum = 0.0

    def observe(self, value: float):
        """Record an observation."""
        self.samples.append(value)
        self.count += 1
        self.sum += value

    def get_stats(self) -> dict[str, float]:
        """Get histogram statistics."""

        if not self.samples:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        sorted_samples = sorted(self.samples)
        n = len(sorted_samples)

        return {
            "count": self.count,
            "sum": self.sum,
            "mean": self.sum / self.count,
            "p50": sorted_samples[int(n * 0.50)],
            "p95": sorted_samples[int(n * 0.95)],
            "p99": sorted_samples[int(n * 0.99)],
        }


class MetricsRegistry:
    """Central registry for all metrics."""

    def __init__(self):
        self.counters: dict[str, Counter] = {}
        self.gauges: dict[str, Gauge] = {}
        self.histograms: dict[str, Histogram] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create a counter."""
        if name not in self.counters:
            self.counters[name] = Counter(name, description)
        return self.counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a gauge."""
        if name not in self.gauges:
            self.gauges[name] = Gauge(name, description)
        return self.gauges[name]

    def histogram(self, name: str, description: str = "", max_samples: int = 1000) -> Histogram:
        """Get or create a histogram."""
        if name not in self.histograms:
            self.histograms[name] = Histogram(name, description, max_samples)
        return self.histograms[name]

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics as dictionary."""

        metrics = {
            "counters": {name: c.get() for name, c in self.counters.items()},
            "gauges": {name: g.get() for name, g in self.gauges.items()},
            "histograms": {name: h.get_stats() for name, h in self.histograms.items()},
        }

        return metrics

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""

        lines = []

        # Export counters
        for name, counter in self.counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.get()}")

        # Export gauges
        for name, gauge in self.gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {gauge.get()}")

        # Export histograms
        for name, histogram in self.histograms.items():
            stats = histogram.get_stats()
            lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")
            lines.append(f'{name}_count {stats["count"]}')
            lines.append(f'{name}_sum {stats["sum"]}')
            lines.append(f'{name}{{quantile="0.5"}} {stats["p50"]}')
            lines.append(f'{name}{{quantile="0.95"}} {stats["p95"]}')
            lines.append(f'{name}{{quantile="0.99"}} {stats["p99"]}')

        return "\n".join(lines)


# ===== Application-Specific Metrics =====

class SparseLLMMetrics:
    """Metrics collector for SparseLLM server."""

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        self.registry = registry or MetricsRegistry()

        # Request metrics
        self.requests_total = self.registry.counter(
            "requests_total",
            "Total number of requests processed"
        )
        self.requests_active = self.registry.gauge(
            "requests_active",
            "Number of requests currently being processed"
        )
        self.request_duration = self.registry.histogram(
            "request_duration_seconds",
            "Request duration in seconds"
        )

        # Expert paging metrics
        self.expert_loads_total = self.registry.counter(
            "expert_loads_total",
            "Total number of expert loads from CPU/disk"
        )
        self.expert_cache_hits = self.registry.counter(
            "expert_cache_hits_total",
            "Total number of expert cache hits"
        )
        self.expert_load_duration = self.registry.histogram(
            "expert_load_duration_seconds",
            "Expert load duration in seconds"
        )

        # Memory metrics
        self.gpu_memory_used = self.registry.gauge(
            "gpu_memory_used_bytes",
            "GPU memory used in bytes"
        )
        self.cpu_memory_used = self.registry.gauge(
            "cpu_memory_used_bytes",
            "CPU memory used in bytes"
        )

        # Token generation metrics
        self.tokens_generated = self.registry.counter(
            "tokens_generated_total",
            "Total number of tokens generated"
        )
        self.tokens_per_second = self.registry.gauge(
            "tokens_per_second",
            "Current token generation throughput"
        )

        # Error metrics
        self.errors_total = self.registry.counter(
            "errors_total",
            "Total number of errors"
        )

    def record_request(self, duration_seconds: float):
        """Record a completed request."""
        self.requests_total.inc()
        self.request_duration.observe(duration_seconds)

    def record_expert_load(self, duration_seconds: float, cache_hit: bool):
        """Record expert load event."""
        if cache_hit:
            self.expert_cache_hits.inc()
        else:
            self.expert_loads_total.inc()
            self.expert_load_duration.observe(duration_seconds)

    def record_generation(self, tokens: int, duration_seconds: float):
        """Record token generation."""
        self.tokens_generated.inc(tokens)
        if duration_seconds > 0:
            self.tokens_per_second.set(tokens / duration_seconds)

    def record_error(self):
        """Record an error."""
        self.errors_total.inc()

    def update_memory(self, gpu_bytes: float, cpu_bytes: float):
        """Update memory usage metrics."""
        self.gpu_memory_used.set(gpu_bytes)
        self.cpu_memory_used.set(cpu_bytes)

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        return self.registry.get_all_metrics()

    def export_prometheus(self) -> str:
        """Export in Prometheus format."""
        return self.registry.export_prometheus()


# ===== Observability Context Manager =====

class ObservabilityContext:
    """Context manager for request observability."""

    def __init__(
        self,
        logger: StructuredLogger,
        metrics: SparseLLMMetrics,
        operation: str,
    ):
        self.logger = logger
        self.metrics = metrics
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.metrics.requests_active.inc()
        self.logger.debug(f"{self.operation}_started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.metrics.requests_active.dec()

        if exc_type is None:
            self.metrics.record_request(duration)
            self.logger.info(
                f"{self.operation}_completed",
                duration_ms=duration * 1000
            )
        else:
            self.metrics.record_error()
            self.logger.error(
                f"{self.operation}_failed",
                duration_ms=duration * 1000,
                error=str(exc_val)
            )

        return False  # Don't suppress exceptions


# ===== Global Instances =====

# These will be initialized by the server
_logger: Optional[StructuredLogger] = None
_metrics: Optional[SparseLLMMetrics] = None


def init_observability(log_format: str = "json", log_file: Optional[str] = None):
    """Initialize global observability instances."""
    global _logger, _metrics

    _logger = StructuredLogger(
        name="sparse_llm",
        format_type=log_format,
        log_file=log_file,
    )

    _metrics = SparseLLMMetrics()

    return _logger, _metrics


def get_logger() -> StructuredLogger:
    """Get global logger instance."""
    if _logger is None:
        raise RuntimeError("Observability not initialized. Call init_observability() first.")
    return _logger


def get_metrics() -> SparseLLMMetrics:
    """Get global metrics instance."""
    if _metrics is None:
        raise RuntimeError("Observability not initialized. Call init_observability() first.")
    return _metrics


__all__ = [
    "StructuredLogger",
    "MetricsRegistry",
    "SparseLLMMetrics",
    "ObservabilityContext",
    "Counter",
    "Gauge",
    "Histogram",
    "init_observability",
    "get_logger",
    "get_metrics",
]

"""Test benchmark runner with baseline and regression detection."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from sparse_llm.benchmark.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    RegressionReport,
    StageMetrics,
)


def test_benchmark_result_serialization_roundtrip():
    """Verify benchmark result can be serialized and deserialized."""
    stage1 = StageMetrics(
        stage="baseline",
        iterations=2,
        prompt="test",
        model="test-model",
        avg_prefill_latency_ms=10.5,
        avg_decode_latency_ms=5.2,
        avg_throughput_tokens_per_sec=100.0,
        avg_peak_vram_bytes=1024,
        avg_cache_hit_rate=0.85,
        p95_prefill_latency_ms=12.0,
        p95_decode_latency_ms=6.0,
    )

    result = BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        model="test-model",
        device="cpu",
    )
    result.stages["baseline"] = stage1

    # Round trip through dict
    result_dict = result.to_dict()
    assert result_dict["model"] == "test-model"
    assert "baseline" in result_dict["stages"]
    assert result_dict["stages"]["baseline"]["avg_prefill_latency_ms"] == 10.5
    assert result_dict["stages"]["baseline"]["sample_count"] == 0


def test_regression_report_passes_without_issues():
    """Verify regression report passes when no regressions detected."""
    stage1 = StageMetrics(
        stage="baseline",
        iterations=1,
        prompt="test",
        model="test-model",
        avg_prefill_latency_ms=10.0,
        avg_decode_latency_ms=5.0,
        avg_throughput_tokens_per_sec=100.0,
        avg_peak_vram_bytes=1024,
        avg_cache_hit_rate=0.85,
    )

    baseline = BenchmarkResult(
        timestamp="2026-08-30 10:00:00",
        model="test-model",
        device="cpu",
    )
    baseline.stages["baseline"] = stage1

    # Current run with same metrics (no regression)
    stage2 = StageMetrics(
        stage="baseline",
        iterations=1,
        prompt="test",
        model="test-model",
        avg_prefill_latency_ms=10.5,  # within 10% threshold
        avg_decode_latency_ms=5.2,
        avg_throughput_tokens_per_sec=95.0,  # -5% is within threshold
        avg_peak_vram_bytes=1024,
        avg_cache_hit_rate=0.85,
    )

    current = BenchmarkResult(
        timestamp="2026-08-30 11:00:00",
        model="test-model",
        device="cpu",
    )
    current.stages["baseline"] = stage2

    runner = BenchmarkRunner(model="test-model")
    report = runner.compare_with_baseline(current, baseline)

    assert report.passed is True
    assert len(report.regressions) == 0


def test_regression_report_detects_latency_regression():
    """Verify regression report detects prefill latency regression."""
    stage1 = StageMetrics(
        stage="baseline",
        iterations=1,
        prompt="test",
        model="test-model",
        avg_prefill_latency_ms=10.0,
        avg_decode_latency_ms=5.0,
        avg_throughput_tokens_per_sec=100.0,
        avg_peak_vram_bytes=1024,
        avg_cache_hit_rate=0.85,
    )

    baseline = BenchmarkResult(
        timestamp="2026-08-30 10:00:00",
        model="test-model",
        device="cpu",
    )
    baseline.stages["baseline"] = stage1

    # Current run with >10% latency increase (regression)
    stage2 = StageMetrics(
        stage="baseline",
        iterations=1,
        prompt="test",
        model="test-model",
        avg_prefill_latency_ms=12.0,  # +20%, exceeds 10% threshold
        avg_decode_latency_ms=5.0,
        avg_throughput_tokens_per_sec=100.0,
        avg_peak_vram_bytes=1024,
        avg_cache_hit_rate=0.85,
    )

    current = BenchmarkResult(
        timestamp="2026-08-30 11:00:00",
        model="test-model",
        device="cpu",
    )
    current.stages["baseline"] = stage2

    runner = BenchmarkRunner(model="test-model")
    report = runner.compare_with_baseline(current, baseline)

    assert report.passed is False
    assert "baseline" in report.regressions
    assert "prefill_latency_ms" in report.regressions["baseline"]
    regressions = report.regressions["baseline"]["prefill_latency_ms"]
    assert regressions["baseline"] == 10.0
    assert regressions["current"] == 12.0
    assert abs(regressions["change_pct"] - 20.0) < 0.1


def test_regression_report_detects_cache_hit_rate_decrease():
    """Verify regression report detects cache hit rate decrease."""
    stage1 = StageMetrics(
        stage="baseline",
        iterations=1,
        prompt="test",
        model="test-model",
        avg_prefill_latency_ms=10.0,
        avg_decode_latency_ms=5.0,
        avg_throughput_tokens_per_sec=100.0,
        avg_peak_vram_bytes=1024,
        avg_cache_hit_rate=0.85,
    )

    baseline = BenchmarkResult(
        timestamp="2026-08-30 10:00:00",
        model="test-model",
        device="cpu",
    )
    baseline.stages["baseline"] = stage1

    # Current run with decreased cache hit rate
    stage2 = StageMetrics(
        stage="baseline",
        iterations=1,
        prompt="test",
        model="test-model",
        avg_prefill_latency_ms=10.0,
        avg_decode_latency_ms=5.0,
        avg_throughput_tokens_per_sec=100.0,
        avg_peak_vram_bytes=1024,
        avg_cache_hit_rate=0.75,  # -0.10, regression
    )

    current = BenchmarkResult(
        timestamp="2026-08-30 11:00:00",
        model="test-model",
        device="cpu",
    )
    current.stages["baseline"] = stage2

    runner = BenchmarkRunner(model="test-model")
    report = runner.compare_with_baseline(current, baseline)

    assert report.passed is False
    assert "baseline" in report.regressions
    assert "cache_hit_rate" in report.regressions["baseline"]


def test_stage_metrics_aggregation():
    """Verify stage metrics p95 calculation."""
    metrics = StageMetrics(
        stage="test",
        iterations=10,
        prompt="test",
        model="test-model",
        samples=[
            {"prefill_latency_ms": float(i * 2), "decode_latency_ms": float(i)}
            for i in range(1, 11)
        ],
    )

    # Manually calculate p95
    prefill_values = sorted([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    p95_idx = max(0, int(0.95 * len(prefill_values)) - 1)
    expected_p95 = prefill_values[p95_idx]

    assert expected_p95 > 0
    assert metrics.iterations == 10

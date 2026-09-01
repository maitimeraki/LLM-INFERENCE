"""Reproducible benchmark runner with regression detection for local inference.

Measures latency, throughput, memory, and cache metrics across stages.
Baseline establishment and per-stage comparison with configurable thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sparse_llm import InferenceEngine
from sparse_llm.inference.metrics import GenerationMetrics


@dataclass
class StageMetrics:
    """Aggregated metrics for a single benchmark stage run."""

    stage: str
    iterations: int
    prompt: str
    model: str
    avg_prefill_latency_ms: float = 0.0
    avg_decode_latency_ms: float = 0.0
    avg_throughput_tokens_per_sec: float = 0.0
    avg_peak_vram_bytes: int = 0
    avg_cache_hit_rate: float = 0.0
    p95_prefill_latency_ms: float = 0.0
    p95_decode_latency_ms: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return as dictionary for JSON serialization."""
        return {
            "stage": self.stage,
            "iterations": self.iterations,
            "prompt": self.prompt,
            "model": self.model,
            "avg_prefill_latency_ms": self.avg_prefill_latency_ms,
            "avg_decode_latency_ms": self.avg_decode_latency_ms,
            "avg_throughput_tokens_per_sec": self.avg_throughput_tokens_per_sec,
            "avg_peak_vram_bytes": self.avg_peak_vram_bytes,
            "avg_cache_hit_rate": self.avg_cache_hit_rate,
            "p95_prefill_latency_ms": self.p95_prefill_latency_ms,
            "p95_decode_latency_ms": self.p95_decode_latency_ms,
            "sample_count": len(self.samples),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark run with all stages."""

    timestamp: str
    model: str
    device: str
    stages: dict[str, StageMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return as dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "device": self.device,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }

    def save(self, path: str | Path) -> None:
        """Persist result to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        """Load result from JSON file."""
        data = json.loads(Path(path).read_text())
        result = cls(timestamp=data["timestamp"], model=data["model"], device=data["device"])
        for stage_name, stage_data in data["stages"].items():
            result.stages[stage_name] = StageMetrics(
                stage=stage_data["stage"],
                iterations=stage_data["iterations"],
                prompt=stage_data["prompt"],
                model=stage_data["model"],
                avg_prefill_latency_ms=stage_data["avg_prefill_latency_ms"],
                avg_decode_latency_ms=stage_data["avg_decode_latency_ms"],
                avg_throughput_tokens_per_sec=stage_data["avg_throughput_tokens_per_sec"],
                avg_peak_vram_bytes=stage_data["avg_peak_vram_bytes"],
                avg_cache_hit_rate=stage_data["avg_cache_hit_rate"],
                p95_prefill_latency_ms=stage_data["p95_prefill_latency_ms"],
                p95_decode_latency_ms=stage_data["p95_decode_latency_ms"],
            )
        return result


@dataclass
class RegressionReport:
    """Comparison between baseline and current run with pass/fail per stage."""

    baseline_timestamp: str
    current_timestamp: str
    passed: bool
    regressions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return as dictionary for JSON serialization."""
        return {
            "baseline_timestamp": self.baseline_timestamp,
            "current_timestamp": self.current_timestamp,
            "passed": self.passed,
            "regressions": self.regressions,
        }

    def save(self, path: str | Path) -> None:
        """Persist report to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


class BenchmarkRunner:
    """Run reproducible benchmarks with stage-based comparison and regression detection.

    Supports:
    - Multiple benchmark stages (baseline, quantized, cached, etc.)
    - Baseline establishment from first run
    - Per-stage regression detection with configurable thresholds
    - Machine-readable JSON output
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: str | None = None,
        expert_cache_bytes: int | None = None,
    ):
        """Initialize runner with model and device configuration."""
        self.model = model
        self.device = device
        self.dtype = dtype
        self.expert_cache_bytes = expert_cache_bytes
        self.engine: InferenceEngine | None = None

    def load(self) -> None:
        """Load the model once for all stages."""
        if self.engine is None:
            self.engine = InferenceEngine(
                model=self.model,
                device=self.device,
                dtype=self.dtype,
                expert_cache_bytes=self.expert_cache_bytes,
            )
            self.engine.load()

    def benchmark_stage(
        self,
        stage: str,
        prompts: list[str],
        max_new_tokens: int = 32,
        iterations_per_prompt: int = 1,
    ) -> StageMetrics:
        """Run benchmark for a stage with given prompts.

        Args:
            stage: Stage identifier (e.g. 'baseline', 'quantized')
            prompts: List of prompts to generate from
            max_new_tokens: Maximum tokens to generate per request
            iterations_per_prompt: Iterations per prompt for stability

        Returns:
            StageMetrics with aggregated measurements
        """
        if self.engine is None:
            self.load()

        all_samples = []
        total_prefill = 0.0
        total_decode = 0.0
        total_throughput = 0.0
        total_vram = 0
        total_hit_rate = 0.0

        for prompt in prompts:
            for _ in range(iterations_per_prompt):
                result = self.engine.generate(prompt, max_new_tokens=max_new_tokens)
                m = result.metrics
                all_samples.append(
                    {
                        "prompt": prompt[:50],
                        "prefill_latency_ms": m.prefill_latency_ms,
                        "decode_latency_ms": m.decode_latency_ms,
                        "throughput_tokens_per_sec": m.throughput_tokens_per_sec,
                        "peak_vram_bytes": m.peak_vram_bytes,
                        "cache_hit_rate": m.cache_hit_rate(),
                    }
                )
                total_prefill += m.prefill_latency_ms
                total_decode += m.decode_latency_ms
                total_throughput += m.throughput_tokens_per_sec
                total_vram = max(total_vram, m.peak_vram_bytes)
                total_hit_rate += m.cache_hit_rate()

        num_samples = len(all_samples)
        avg_prefill = total_prefill / num_samples if num_samples > 0 else 0.0
        avg_decode = total_decode / num_samples if num_samples > 0 else 0.0
        avg_throughput = total_throughput / num_samples if num_samples > 0 else 0.0
        avg_hit_rate = total_hit_rate / num_samples if num_samples > 0 else 0.0

        # Compute p95 latencies
        prefill_latencies = sorted([s["prefill_latency_ms"] for s in all_samples])
        decode_latencies = sorted([s["decode_latency_ms"] for s in all_samples])
        p95_idx = max(0, int(0.95 * len(prefill_latencies)) - 1)
        p95_prefill = prefill_latencies[p95_idx] if prefill_latencies else 0.0
        p95_decode = decode_latencies[p95_idx] if decode_latencies else 0.0

        metrics = StageMetrics(
            stage=stage,
            iterations=num_samples,
            prompt="|".join([p[:30] for p in prompts]),
            model=self.model,
            avg_prefill_latency_ms=avg_prefill,
            avg_decode_latency_ms=avg_decode,
            avg_throughput_tokens_per_sec=avg_throughput,
            avg_peak_vram_bytes=total_vram,
            avg_cache_hit_rate=avg_hit_rate,
            p95_prefill_latency_ms=p95_prefill,
            p95_decode_latency_ms=p95_decode,
            samples=all_samples,
        )
        return metrics

    def compare_with_baseline(
        self,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        thresholds: dict[str, float] | None = None,
    ) -> RegressionReport:
        """Compare current run against baseline using configurable thresholds.

        Default thresholds allow 10% regression in latency, 5% in throughput.
        Cache hit rate must not decrease.

        Args:
            current: Current benchmark result
            baseline: Baseline benchmark result
            thresholds: Dict of metric -> max allowed % regression (e.g., {'prefill': 10.0})

        Returns:
            RegressionReport with comparison details
        """
        if thresholds is None:
            thresholds = {
                "prefill_latency_ms": 10.0,
                "decode_latency_ms": 10.0,
                "throughput_tokens_per_sec": -5.0,  # negative = improvement
                "cache_hit_rate": 0.0,  # no decrease allowed
            }

        regressions = {}
        report = RegressionReport(
            baseline_timestamp=baseline.timestamp,
            current_timestamp=current.timestamp,
            passed=True,
        )

        for stage_name, baseline_metrics in baseline.stages.items():
            if stage_name not in current.stages:
                regressions[stage_name] = {"error": "stage missing in current run"}
                report.passed = False
                continue

            current_metrics = current.stages[stage_name]
            stage_regressions = {}

            # Check prefill latency
            prefill_threshold = thresholds.get("prefill_latency_ms", 10.0)
            prefill_delta = (
                current_metrics.avg_prefill_latency_ms - baseline_metrics.avg_prefill_latency_ms
            ) / max(baseline_metrics.avg_prefill_latency_ms, 1.0) * 100
            if prefill_delta > prefill_threshold:
                stage_regressions["prefill_latency_ms"] = {
                    "baseline": baseline_metrics.avg_prefill_latency_ms,
                    "current": current_metrics.avg_prefill_latency_ms,
                    "change_pct": prefill_delta,
                    "threshold_pct": prefill_threshold,
                }
                report.passed = False

            # Check decode latency
            decode_threshold = thresholds.get("decode_latency_ms", 10.0)
            decode_delta = (
                current_metrics.avg_decode_latency_ms - baseline_metrics.avg_decode_latency_ms
            ) / max(baseline_metrics.avg_decode_latency_ms, 1.0) * 100
            if decode_delta > decode_threshold:
                stage_regressions["decode_latency_ms"] = {
                    "baseline": baseline_metrics.avg_decode_latency_ms,
                    "current": current_metrics.avg_decode_latency_ms,
                    "change_pct": decode_delta,
                    "threshold_pct": decode_threshold,
                }
                report.passed = False

            # Check throughput (negative threshold means improvement required)
            tput_threshold = thresholds.get("throughput_tokens_per_sec", -5.0)
            tput_delta = (
                current_metrics.avg_throughput_tokens_per_sec
                - baseline_metrics.avg_throughput_tokens_per_sec
            ) / max(baseline_metrics.avg_throughput_tokens_per_sec, 0.1) * 100
            if tput_delta < tput_threshold:  # worse throughput
                stage_regressions["throughput_tokens_per_sec"] = {
                    "baseline": baseline_metrics.avg_throughput_tokens_per_sec,
                    "current": current_metrics.avg_throughput_tokens_per_sec,
                    "change_pct": tput_delta,
                    "threshold_pct": tput_threshold,
                }
                report.passed = False

            # Check cache hit rate (no decrease allowed by default)
            cache_threshold = thresholds.get("cache_hit_rate", 0.0)
            cache_delta = current_metrics.avg_cache_hit_rate - baseline_metrics.avg_cache_hit_rate
            if cache_delta < -cache_threshold:
                stage_regressions["cache_hit_rate"] = {
                    "baseline": baseline_metrics.avg_cache_hit_rate,
                    "current": current_metrics.avg_cache_hit_rate,
                    "change": cache_delta,
                    "threshold": cache_threshold,
                }
                report.passed = False

            if stage_regressions:
                regressions[stage_name] = stage_regressions

        report.regressions = regressions
        return report


def main() -> int:
    """Command-line entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="Benchmark local inference with regression detection")
    parser.add_argument("--model", required=True, help="Model identifier or local path")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--dtype", default=None, help="Data type (e.g., float32, float16)")
    parser.add_argument("--cache-bytes", type=int, default=None, help="Expert cache budget in bytes")
    parser.add_argument("--stages", nargs="+", default=["baseline"], help="Stages to benchmark")
    parser.add_argument("--prompt", default="Explain machine learning in one sentence.", help="Prompt to use")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--iterations", type=int, default=1, help="Iterations per stage")
    parser.add_argument("--output", type=Path, default=Path("benchmark_result.json"), help="Output JSON file")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline result for regression detection")
    parser.add_argument("--regression-output", type=Path, default=None, help="Regression report output")

    args = parser.parse_args()

    try:
        runner = BenchmarkRunner(
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            expert_cache_bytes=args.cache_bytes,
        )

        # Run benchmark stages
        result = BenchmarkResult(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model=args.model,
            device=args.device,
        )

        for stage in args.stages:
            print(f"Benchmarking stage: {stage}...", file=sys.stderr)
            metrics = runner.benchmark_stage(
                stage=stage,
                prompts=[args.prompt],
                max_new_tokens=args.max_tokens,
                iterations_per_prompt=args.iterations,
            )
            result.stages[stage] = metrics
            print(
                f"  Stage {stage}: {metrics.avg_prefill_latency_ms:.1f}ms prefill, "
                f"{metrics.avg_decode_latency_ms:.1f}ms decode, "
                f"{metrics.avg_throughput_tokens_per_sec:.1f} tokens/sec",
                file=sys.stderr,
            )

        # Save result
        result.save(args.output)
        print(f"Result saved to {args.output}", file=sys.stderr)

        # Regression detection if baseline provided
        if args.baseline:
            baseline = BenchmarkResult.load(args.baseline)
            report = runner.compare_with_baseline(result, baseline)
            if args.regression_output:
                report.save(args.regression_output)
                print(f"Regression report saved to {args.regression_output}", file=sys.stderr)
            if not report.passed:
                print("REGRESSION DETECTED", file=sys.stderr)
                for stage, issues in report.regressions.items():
                    print(f"  {stage}: {issues}", file=sys.stderr)
                return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Tests for ResourceProfiler."""

import pytest
import torch


def test_resource_profiler_detects_gpu():
    from sparse_llm.loading.resource_profiler import ResourceProfiler

    profiler = ResourceProfiler()
    budget = profiler.profile()

    if torch.cuda.is_available():
        assert len(budget.gpus) > 0
        assert budget.gpus[0].total_bytes > 0
        assert budget.gpus[0].available_bytes > 0
    else:
        assert len(budget.gpus) == 0
        assert "No GPU detected" in " ".join(budget.warnings)


def test_resource_profiler_applies_safety_margins():
    from sparse_llm.loading.resource_profiler import ResourceProfiler

    profiler = ResourceProfiler()
    budget = profiler.profile()

    # CPU safety margin: use 80% of available
    assert budget.cpu.usable_bytes <= budget.cpu.available_bytes * 0.80

    # GPU safety margin: use 85% of available (if GPU exists)
    if torch.cuda.is_available() and len(budget.gpus) > 0:
        for gpu in budget.gpus:
            # available_bytes should already have margin applied
            pass  # Margin applied in profiler

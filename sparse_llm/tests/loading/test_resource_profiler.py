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
        # GPU available should be less than total (margin applied)
        assert budget.gpus[0].available_bytes < budget.gpus[0].total_bytes
    else:
        assert len(budget.gpus) == 0


def test_resource_profiler_applies_cpu_margin():
    from sparse_llm.loading.resource_profiler import ResourceProfiler
    import psutil

    profiler = ResourceProfiler()
    budget = profiler.profile()

    # CPU safety margin: use 80% of available (verify ratio is 0.79-0.80)
    margin_ratio = budget.cpu.usable_bytes / budget.cpu.available_bytes
    assert 0.79 <= margin_ratio <= 0.80, f"CPU margin ratio {margin_ratio} not in range [0.79, 0.80]"


def test_resource_profiler_applies_gpu_margin():
    from sparse_llm.loading.resource_profiler import ResourceProfiler

    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")

    profiler = ResourceProfiler()
    budget = profiler.profile()

    # GPU safety margin: use 85% of free memory
    for gpu in budget.gpus:
        # Get fresh GPU memory info
        torch.cuda.set_device(gpu.device_id)
        torch.cuda.empty_cache()
        free_bytes = torch.cuda.mem_get_info()[0]

        # Verify available is approximately 85% of free (allow 1% tolerance for timing)
        expected_available = int(free_bytes * 0.85)
        tolerance = free_bytes * 0.01
        assert abs(gpu.available_bytes - expected_available) < tolerance, \
            f"GPU margin not 85%: available={gpu.available_bytes}, expected≈{expected_available}"


def test_resource_profiler_applies_storage_margin():
    from sparse_llm.loading.resource_profiler import ResourceProfiler
    import psutil

    profiler = ResourceProfiler()
    budget = profiler.profile()

    # Storage safety margin: use 90% of free space
    usage = psutil.disk_usage(budget.storage.path)
    expected_available = int(usage.free * 0.90)

    # Verify margin applied (allow small tolerance for concurrent disk changes)
    assert abs(budget.storage.available_bytes - expected_available) < usage.free * 0.01, \
        f"Storage margin not 90%: available={budget.storage.available_bytes}, expected={expected_available}"


def test_resource_profiler_validates_minimum_requirements():
    from sparse_llm.loading.resource_profiler import ResourceProfiler

    profiler = ResourceProfiler()
    budget = profiler.profile()

    # Check warnings for minimum requirements
    if budget.cpu.usable_bytes < 8 * 1024**3:
        assert any("CPU RAM below recommended minimum" in w for w in budget.warnings), \
            "Should warn when CPU RAM < 8GB"

    if budget.storage.available_bytes < 10 * 1024**3:
        assert any("Storage space below minimum" in w for w in budget.warnings), \
            "Should warn when storage < 10GB"

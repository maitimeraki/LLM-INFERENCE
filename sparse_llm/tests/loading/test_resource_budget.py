def test_resource_budget_creation():
    from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo

    gpu = GPUInfo(device_id=0, total_bytes=24*1024**3, available_bytes=22*1024**3,
                  compute_capability=(8, 0), name="NVIDIA A100")
    cpu = CPUInfo(total_bytes=64*1024**3, available_bytes=48*1024**3, usable_bytes=38*1024**3)
    storage = StorageInfo(path="/root/.cache/sparse_llm", available_bytes=500*1024**3,
                         is_ssd=True, estimated_bandwidth_mbps=1200)

    budget = ResourceBudget(gpus=[gpu], cpu=cpu, storage=storage)

    assert budget.total_gpu_bytes == 22*1024**3
    assert budget.total_cpu_bytes == 38*1024**3
    assert len(budget.warnings) == 0


def test_resource_budget_warnings_low_cpu():
    from sparse_llm.loading.resource_budget import ResourceBudget, GPUInfo, CPUInfo, StorageInfo

    gpu = GPUInfo(device_id=0, total_bytes=24*1024**3, available_bytes=22*1024**3,
                  compute_capability=(8, 0), name="NVIDIA A100")
    cpu = CPUInfo(total_bytes=6*1024**3, available_bytes=5*1024**3, usable_bytes=4*1024**3)  # Low RAM
    storage = StorageInfo(path="/root/.cache/sparse_llm", available_bytes=500*1024**3,
                         is_ssd=True, estimated_bandwidth_mbps=1200)

    budget = ResourceBudget(gpus=[gpu], cpu=cpu, storage=storage)

    assert any("CPU RAM below recommended minimum" in w for w in budget.warnings)

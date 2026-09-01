import threading
from types import SimpleNamespace

import pytest
import torch

from sparse_llm.cache import ExpertCache
from sparse_llm.inference.memory import MemoryAdmissionError, MemoryPlanner
from sparse_llm.inference.pager import ExpertPager
from sparse_llm.models.paging import (
    PagingCapabilities,
    PagingValidationResult,
    PagedMoELayer,
    RouterSelection,
)
from sparse_llm.storage.checkpoint_index import CheckpointIndex, ExpertTensorMapping


safetensors = pytest.importorskip("safetensors.torch")


def write_fixture_checkpoint(tmp_path):
    tensors = {
        "shared.bias": torch.tensor([0.25, -0.5]),
        "layer.0.experts.0.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "layer.0.experts.0.bias": torch.tensor([0.0, 0.0]),
        "layer.0.experts.1.weight": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        "layer.0.experts.1.bias": torch.tensor([0.0, 0.0]),
        "layer.1.experts.0.weight": torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
    }
    safetensors.save_file(tensors, str(tmp_path / "model.safetensors"))
    return tensors


def fixture_mapper(name):
    parts = name.split(".")
    if len(parts) >= 5 and parts[0] == "layer" and parts[2] == "experts":
        return ExpertTensorMapping((int(parts[1]), int(parts[3])), name)
    return None


def test_checkpoint_index_maps_exact_layer_aware_expert_tensors(tmp_path):
    tensors = write_fixture_checkpoint(tmp_path)

    index = CheckpointIndex.from_directory(tmp_path, fixture_mapper)

    assert index.expert_keys == ((0, 0), (0, 1), (1, 0))
    assert index.shared_tensor_names == ("shared.bias",)
    assert index.expert((0, 1)).byte_size == tensors["layer.0.experts.1.weight"].nbytes + tensors[
        "layer.0.experts.1.bias"
    ].nbytes
    loaded = index.load_expert((0, 1))
    assert torch.equal(loaded["layer.0.experts.1.weight"], tensors["layer.0.experts.1.weight"])


def test_checkpoint_index_rejects_unsafe_checkpoint_by_default(tmp_path):
    torch.save({"weight": torch.ones(1)}, tmp_path / "pytorch_model.bin")

    with pytest.raises(ValueError, match="safetensors"):
        CheckpointIndex.from_directory(tmp_path, fixture_mapper)


def test_checkpoint_index_rejects_missing_expected_expert(tmp_path):
    write_fixture_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="missing.*\(2, 0\)"):
        CheckpointIndex.from_directory(tmp_path, fixture_mapper, expected_keys={(0, 0), (2, 0)})


def test_memory_planner_rejects_shared_footprint_before_expert_loading():
    planner = MemoryPlanner(
        capacity_bytes=100,
        expert_cache_bytes=80,
        kv_cache_bytes=10,
        safety_margin_bytes=10,
    )

    plan = planner.plan(shared_bytes=81, routed_group_bytes=1)

    assert not plan.can_admit
    assert "shared" in plan.rejection_reason
    with pytest.raises(MemoryAdmissionError, match="shared"):
        planner.require(plan)


def test_memory_planner_rejects_routed_group_larger_than_budget():
    planner = MemoryPlanner(capacity_bytes=100, expert_cache_bytes=30, safety_margin_bytes=10)

    plan = planner.plan(shared_bytes=20, routed_group_bytes=31)

    assert not plan.can_admit
    assert "routed" in plan.rejection_reason


def test_pager_loads_only_requested_expert_and_reports_source(tmp_path):
    tensors = write_fixture_checkpoint(tmp_path)
    index = CheckpointIndex.from_directory(tmp_path, fixture_mapper)
    pager = ExpertPager(
        index,
        ExpertCache(max_experts=2, max_bytes=64),
        device="cpu",
    )

    with pager.lease((0, 1)) as loaded:
        assert set(loaded.tensors) == {
            "layer.0.experts.1.weight",
            "layer.0.experts.1.bias",
        }
        assert loaded.source_tier == "ssd"
        assert loaded.byte_size == tensors["layer.0.experts.1.weight"].nbytes + tensors[
            "layer.0.experts.1.bias"
        ].nbytes

    stats = pager.request_stats()
    assert stats["cache_misses"] == 1
    assert stats["cache_hits"] == 0
    assert stats["routed_keys"] == []


def test_pager_keeps_leased_entry_pinned_during_capacity_pressure(tmp_path):
    write_fixture_checkpoint(tmp_path)
    index = CheckpointIndex.from_directory(tmp_path, fixture_mapper)
    cache = ExpertCache(max_experts=2, max_bytes=24)
    pager = ExpertPager(index, cache, device="cpu")

    with pager.lease((0, 0)):
        with pytest.raises(MemoryError, match="pinned"):
            with pager.lease((0, 1)):
                pass
        assert cache.contains((0, 0))


def test_pager_deduplicates_concurrent_loads(tmp_path):
    write_fixture_checkpoint(tmp_path)
    index = CheckpointIndex.from_directory(tmp_path, fixture_mapper)
    pager = ExpertPager(index, ExpertCache(max_experts=2, max_bytes=64), device="cpu")
    original = index.load_expert
    calls = []
    started = threading.Event()
    release = threading.Event()

    def load(key):
        calls.append(key)
        started.set()
        release.wait(timeout=2)
        return original(key)

    index.load_expert = load
    results = []

    def run():
        with pager.lease((0, 0)) as expert:
            results.append(expert.tensors["layer.0.experts.0.weight"])

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert started.wait(timeout=2)
    release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert calls == [(0, 0)]
    assert len(results) == 2
    assert pager.request_stats()["cache_misses"] == 2


def test_paged_moe_layer_matches_reference_dispatch(tmp_path):
    tensors = write_fixture_checkpoint(tmp_path)
    index = CheckpointIndex.from_directory(tmp_path, fixture_mapper)
    pager = ExpertPager(index, ExpertCache(max_experts=2, max_bytes=64), device="cpu")

    def router(hidden):
        indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        return RouterSelection(indices, probabilities)

    def expert_forward(key, inputs, loaded):
        weight = loaded.tensors[f"layer.{key[0]}.experts.{key[1]}.weight"]
        bias_name = f"layer.{key[0]}.experts.{key[1]}.bias"
        bias = loaded.tensors.get(bias_name, torch.zeros(weight.shape[0]))
        return inputs @ weight.t() + bias

    layer = PagedMoELayer(
        router=router,
        pager=pager,
        layer_index=0,
        num_experts=2,
        top_k=2,
        expert_forward=expert_forward,
    )
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actual = layer(hidden)
    expected = torch.stack(
        (
            0.25 * (hidden[0] @ tensors["layer.0.experts.0.weight"].t())
            + 0.75 * (hidden[0] @ tensors["layer.0.experts.1.weight"].t()),
            0.6 * (hidden[1] @ tensors["layer.0.experts.1.weight"].t())
            + 0.4 * (hidden[1] @ tensors["layer.0.experts.0.weight"].t()),
        )
    )

    assert torch.allclose(actual, expected)
    assert pager.request_stats()["routed_keys"] == [(0, 0), (0, 1)]


def test_paging_contract_reports_failed_validation_as_data():
    capabilities = PagingCapabilities(
        architecture_id="fixture-moe",
        adapter_version="1",
        num_layers=1,
        num_experts=2,
        top_k=2,
        validation_passed=False,
        failure_reasons=("router output shape is unknown",),
    )
    result = PagingValidationResult.from_capabilities(capabilities)

    assert not result.eligible
    assert result.failure_reasons == ("router output shape is unknown",)
    assert result.to_dict()["eligible"] is False


def test_generic_adapter_has_no_paging_capability():
    from sparse_llm.models.adapters import DevicePolicy, TransformersCausalLMAdapter

    adapter = TransformersCausalLMAdapter("local/dense", policy=DevicePolicy(device="cpu"))

    assert adapter.paging_capabilities is None
    assert not adapter.validate_paging().eligible

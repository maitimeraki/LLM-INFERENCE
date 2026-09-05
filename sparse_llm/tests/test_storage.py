import pytest
import torch

from sparse_llm.storage import LocalSSDStorage


def test_save_load_exists_and_layer_isolation(tmp_path):
    storage = LocalSSDStorage(tmp_path)
    first = torch.tensor([1.0, 2.0])
    second = torch.tensor([3.0])

    storage.save_expert((0, 4), first)
    storage.save_expert((1, 4), second)

    assert storage.exists((0, 4))
    assert storage.exists((1, 4))
    assert not storage.exists((2, 4))
    assert torch.equal(storage.load_expert((0, 4)), first)
    assert torch.equal(storage.load_expert((1, 4)), second)


def test_missing_expert_has_actionable_error(tmp_path):
    storage = LocalSSDStorage(tmp_path)

    with pytest.raises(FileNotFoundError, match="layer 2.*expert 9"):
        storage.load_expert((2, 9))


def test_save_detaches_and_loads_on_cpu(tmp_path):
    storage = LocalSSDStorage(tmp_path)
    source = torch.tensor([1.0], requires_grad=True)

    storage.save_expert((3, 7), source)
    loaded = storage.load_expert((3, 7))

    assert loaded.device.type == "cpu"
    assert not loaded.requires_grad
    assert torch.equal(loaded, source.detach())


def test_integer_compatibility_uses_legacy_filename(tmp_path):
    storage = LocalSSDStorage(tmp_path)
    weights = torch.tensor([5.0])

    storage.save_expert(3, weights)

    assert (tmp_path / "expert_000003.pt").exists()
    assert torch.equal(storage.load_expert(3), weights)


def test_integer_with_layer_id_is_layer_aware(tmp_path):
    storage = LocalSSDStorage(tmp_path)
    weights = torch.tensor([6.0])

    storage.save_expert(3, weights, layer_id=2)

    assert (tmp_path / "layer_0002_expert_0003.pt").exists()
    assert torch.equal(storage.load_expert(3, layer_id=2), weights)


def test_invalid_keys_are_rejected(tmp_path):
    storage = LocalSSDStorage(tmp_path)

    with pytest.raises(ValueError, match="ExpertKey"):
        storage.exists((1, 2, 3))
    with pytest.raises(ValueError, match="non-negative"):
        storage.exists((-1, 2))
    with pytest.raises(ValueError, match="integer"):
        storage.exists((1, True))


def test_compatibility_reexports_are_canonical():
    from sparse_llm.storage.backend import LocalSSDStorage as BackendStorage
    from sparse_llm.storage.base import LocalSSDStorage as BaseStorage
    from sparse_llm.storage.local_ssd import LocalSSDStorage as LocalStorage

    assert BaseStorage is BackendStorage
    assert LocalStorage is BackendStorage


def test_temporary_write_is_cleaned_after_save_failure(tmp_path, monkeypatch):
    storage = LocalSSDStorage(tmp_path)

    def fail_save(*args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="save failed"):
        storage.save_expert((0, 0), torch.tensor([1.0]))

    assert list(tmp_path.iterdir()) == []


def test_storage_metadata_compatibility_export_and_network_import():
    from sparse_llm.storage import ExpertMetadata
    from sparse_llm.storage.network import NetworkStorage

    assert NetworkStorage is not None
    metadata = ExpertMetadata(
        expert_id=1,
        layer_idx=2,
        size_bytes=4,
        dtype="float32",
        quantized=False,
    )
    assert metadata.expert_id == 1


def test_load_does_not_fallback_for_unrelated_type_error(tmp_path, monkeypatch):
    storage = LocalSSDStorage(tmp_path)
    storage.save_expert((0, 0), torch.tensor([1.0]))
    calls = []

    def fail_load(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("tensor payload is invalid")

    monkeypatch.setattr(torch, "load", fail_load)
    with pytest.raises(TypeError, match="tensor payload is invalid"):
        storage.load_expert((0, 0))

    assert len(calls) == 1
    assert calls[0]["weights_only"] is True


def test_load_falls_back_when_weights_only_is_unsupported(tmp_path, monkeypatch):
    storage = LocalSSDStorage(tmp_path)
    expected = torch.tensor([1.0])
    storage.save_expert((0, 0), expected)
    calls = []

    def load_without_weights_only(path, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("load() got an unexpected keyword argument 'weights_only'")
        return expected

    monkeypatch.setattr(torch, "load", load_without_weights_only)
    assert torch.equal(storage.load_expert((0, 0)), expected)
    assert calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    ]


def test_bare_integer_ignores_noncanonical_layer_candidates(tmp_path):
    storage = LocalSSDStorage(tmp_path)
    expected = torch.tensor([1.0])
    storage.save_expert((2, 3), expected)
    (tmp_path / "layer_bad_expert_0003.pt").touch()
    (tmp_path / "layer_02_expert_0003.pt").touch()

    assert storage.exists(3)
    assert torch.equal(storage.load_expert(3), expected)


def test_bare_integer_layer_ambiguity_is_consistent_for_exists_and_load(tmp_path):
    storage = LocalSSDStorage(tmp_path)
    storage.save_expert((0, 4), torch.tensor([1.0]))
    storage.save_expert((1, 4), torch.tensor([2.0]))

    with pytest.raises(ValueError, match="ambiguous"):
        storage.exists(4)
    with pytest.raises(ValueError, match="ambiguous"):
        storage.load_expert(4)


def test_bare_integer_ambiguity_is_documented_by_method_contract():
    assert "ambiguous" in (LocalSSDStorage.exists.__doc__ or "")
    assert "ambiguous" in (LocalSSDStorage.load_expert.__doc__ or "")

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

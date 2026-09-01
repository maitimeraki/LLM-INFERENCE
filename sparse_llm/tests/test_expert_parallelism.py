"""Tests for multi-GPU expert parallelism."""

import pytest
import torch

from sparse_llm.models.expert_parallelism import (
    ExpertDeviceMapping,
    ExpertParallelismManager,
)


class TestExpertDeviceMapping:
    """Test ExpertDeviceMapping dataclass."""

    def test_creation_replicated(self):
        devices = (torch.device("cpu"), torch.device("cpu"))
        mapping = ExpertDeviceMapping(
            expert_key=(0, 0), devices=devices, strategy="replicated"
        )
        assert mapping.expert_key == (0, 0)
        assert mapping.devices == devices
        assert mapping.strategy == "replicated"

    def test_creation_sharded(self):
        devices = (torch.device("cpu"),)
        mapping = ExpertDeviceMapping(
            expert_key=(1, 2), devices=devices, strategy="sharded"
        )
        assert mapping.expert_key == (1, 2)
        assert mapping.strategy == "sharded"

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValueError, match="strategy must be"):
            ExpertDeviceMapping(
                expert_key=(0, 0),
                devices=(torch.device("cpu"),),
                strategy="invalid",
            )

    def test_empty_devices_rejected(self):
        with pytest.raises(ValueError, match="devices cannot be empty"):
            ExpertDeviceMapping(expert_key=(0, 0), devices=(), strategy="replicated")


class TestExpertParallelismManager:
    """Test ExpertParallelismManager."""

    def test_initialization_replicated(self):
        manager = ExpertParallelismManager(num_gpus=2, strategy="replicated")
        assert manager.num_gpus == 2
        assert manager.strategy == "replicated"
        assert len(manager.devices) == 2

    def test_initialization_sharded(self):
        manager = ExpertParallelismManager(num_gpus=4, strategy="sharded")
        assert manager.num_gpus == 4
        assert manager.strategy == "sharded"

    def test_custom_device_ids(self):
        manager = ExpertParallelismManager(num_gpus=2, device_ids=[2, 3])
        assert manager.device_ids == [2, 3]

    def test_invalid_num_gpus(self):
        with pytest.raises(ValueError, match="num_gpus must be a positive integer"):
            ExpertParallelismManager(num_gpus=0)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="strategy must be"):
            ExpertParallelismManager(strategy="invalid")

    def test_register_expert_replicated(self):
        manager = ExpertParallelismManager(num_gpus=2, strategy="replicated")
        manager.register_expert((0, 0))
        assert (0, 0) in manager.expert_placement
        devices = manager.get_devices((0, 0))
        assert len(devices) == 2

    def test_register_expert_sharded(self):
        manager = ExpertParallelismManager(num_gpus=3, strategy="sharded")
        manager.register_expert((0, 0), gpu_idx=1)
        assert (0, 0) in manager.expert_placement
        devices = manager.get_devices((0, 0))
        assert len(devices) == 1

    def test_sharded_auto_assignment(self):
        manager = ExpertParallelismManager(num_gpus=3, strategy="sharded")
        manager.register_expert((0, 0))
        manager.register_expert((0, 1))
        manager.register_expert((0, 2))
        # Should round-robin
        gpu_idx_0 = manager.expert_to_gpu_idx[(0, 0)]
        gpu_idx_1 = manager.expert_to_gpu_idx[(0, 1)]
        gpu_idx_2 = manager.expert_to_gpu_idx[(0, 2)]
        assert {gpu_idx_0, gpu_idx_1, gpu_idx_2} == {0, 1, 2}

    def test_get_primary_device(self):
        manager = ExpertParallelismManager(num_gpus=2, strategy="replicated")
        manager.register_expert((0, 0))
        primary = manager.get_primary_device((0, 0))
        assert str(primary).startswith("cuda:")

    def test_transfer_expert_cpu(self):
        # Use CPU-only for test
        manager = ExpertParallelismManager(num_gpus=1, device_ids=[0])
        manager.devices = [torch.device("cpu")]
        manager.register_expert((0, 0))
        tensor = torch.randn(10, 10)
        result = manager.transfer_expert(tensor, (0, 0), target_device=torch.device("cpu"))
        assert result.device.type == "cpu"

    def test_transfer_expert_invalid_target(self):
        manager = ExpertParallelismManager(num_gpus=1, strategy="sharded", device_ids=[0])
        manager.register_expert((0, 0), gpu_idx=0)
        tensor = torch.randn(10, 10)
        # Try to transfer to a device that wasn't registered
        invalid_device = torch.device("cpu") if manager.devices[0].type != "cpu" else torch.device("meta")
        with pytest.raises(ValueError, match="not in allowed devices"):
            manager.transfer_expert(tensor, (0, 0), target_device=invalid_device)

    def test_distribute_tensor_replicated(self):
        # Test that replicated strategy distributes to all devices
        manager = ExpertParallelismManager(num_gpus=1, strategy="replicated", device_ids=[0])
        manager.devices = [torch.device("cpu")]
        manager.register_expert((0, 0))
        tensor = torch.randn(10, 10)
        dist = manager.distribute_tensor(tensor, (0, 0))
        assert len(dist) >= 1
        for device, t in dist.items():
            assert t.device == device

    def test_distribute_tensor_sharded(self):
        manager = ExpertParallelismManager(num_gpus=2, strategy="sharded")
        manager.devices = [torch.device("cpu"), torch.device("cpu")]
        manager.register_expert((0, 0), gpu_idx=0)
        tensor = torch.randn(10, 10)
        dist = manager.distribute_tensor(tensor, (0, 0))
        assert len(dist) == 1

    def test_placement_summary(self):
        manager = ExpertParallelismManager(num_gpus=2, strategy="sharded")
        manager.register_expert((0, 0))
        manager.register_expert((0, 1), gpu_idx=1)
        summary = manager.get_expert_placement_summary()
        assert summary["strategy"] == "sharded"
        assert summary["num_gpus"] == 2
        assert summary["registered_experts"] == 2

    def test_expert_not_registered(self):
        manager = ExpertParallelismManager(num_gpus=2)
        with pytest.raises(KeyError):
            manager.get_devices((99, 99))

    def test_invalid_gpu_idx_sharded(self):
        manager = ExpertParallelismManager(num_gpus=2, strategy="sharded")
        with pytest.raises(ValueError, match="gpu_idx must be in"):
            manager.register_expert((0, 0), gpu_idx=5)

    def test_register_same_expert_twice(self):
        manager = ExpertParallelismManager(num_gpus=2, strategy="replicated")
        manager.register_expert((0, 0))
        manager.register_expert((0, 0))  # Should be no-op
        assert len(manager.expert_placement) == 1

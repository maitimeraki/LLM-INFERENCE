"""Multi-GPU expert distribution for single-machine parallelism.

Distributes experts across available GPUs via replicated (all experts on all
GPUs) or sharded (each expert on exactly one GPU) strategies. Tracks device
placement and manages cross-device transfers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch

ExpertKey = tuple[int, int]


@dataclass(frozen=True)
class ExpertDeviceMapping:
    """Immutable record of which device(s) hold an expert."""

    expert_key: ExpertKey
    devices: tuple[torch.device, ...] = field(default_factory=tuple)
    strategy: str = "replicated"  # "replicated" or "sharded"

    def __post_init__(self):
        if self.strategy not in ("replicated", "sharded"):
            raise ValueError(f"strategy must be 'replicated' or 'sharded', got {self.strategy!r}")
        if not self.devices:
            raise ValueError("devices cannot be empty")


class ExpertParallelismManager:
    """Manages multi-GPU expert distribution and cross-device transfers."""

    def __init__(
        self,
        num_gpus: int = 1,
        strategy: str = "replicated",
        device_ids: Optional[list[int]] = None,
    ):
        """Initialize parallelism manager.

        Args:
            num_gpus: Number of GPUs to use.
            strategy: "replicated" (all experts on all GPUs) or "sharded"
              (each expert on exactly one GPU).
            device_ids: Explicit GPU device IDs. If None, uses [0, 1, ..., num_gpus-1].

        Raises:
            ValueError: If strategy is invalid or num_gpus < 1.
        """
        if strategy not in ("replicated", "sharded"):
            raise ValueError(f"strategy must be 'replicated' or 'sharded', got {strategy!r}")
        if not isinstance(num_gpus, int) or num_gpus < 1:
            raise ValueError("num_gpus must be a positive integer")

        self.num_gpus = num_gpus
        self.strategy = strategy
        self.device_ids = device_ids or list(range(num_gpus))
        self.devices = [torch.device(f"cuda:{i}") for i in self.device_ids]
        self.expert_placement: dict[ExpertKey, ExpertDeviceMapping] = {}
        self.expert_to_gpu_idx: dict[ExpertKey, int] = {}

    def register_expert(self, expert_key: ExpertKey, gpu_idx: Optional[int] = None) -> None:
        """Register an expert's device placement.

        Args:
            expert_key: (layer_id, expert_id) tuple.
            gpu_idx: For sharded strategy, which GPU holds this expert.
              For replicated, ignored (all GPUs hold it).

        Raises:
            ValueError: If sharded strategy requires explicit gpu_idx.
        """
        if expert_key in self.expert_placement:
            return

        if self.strategy == "replicated":
            mapping = ExpertDeviceMapping(
                expert_key=expert_key, devices=tuple(self.devices), strategy="replicated"
            )
        else:  # sharded
            if gpu_idx is None:
                gpu_idx = len(self.expert_to_gpu_idx) % self.num_gpus
            if not isinstance(gpu_idx, int) or gpu_idx < 0 or gpu_idx >= self.num_gpus:
                raise ValueError(f"gpu_idx must be in [0, {self.num_gpus - 1}], got {gpu_idx}")
            mapping = ExpertDeviceMapping(
                expert_key=expert_key, devices=(self.devices[gpu_idx],), strategy="sharded"
            )
            self.expert_to_gpu_idx[expert_key] = gpu_idx

        self.expert_placement[expert_key] = mapping

    def get_devices(self, expert_key: ExpertKey) -> tuple[torch.device, ...]:
        """Get the device(s) where an expert is stored.

        Args:
            expert_key: (layer_id, expert_id) tuple.

        Returns:
            Tuple of torch.device objects.

        Raises:
            KeyError: If expert not registered.
        """
        mapping = self.expert_placement[expert_key]
        return mapping.devices

    def get_primary_device(self, expert_key: ExpertKey) -> torch.device:
        """Get the primary device for an expert (first in devices).

        Args:
            expert_key: (layer_id, expert_id) tuple.

        Returns:
            torch.device object.

        Raises:
            KeyError: If expert not registered.
        """
        return self.expert_placement[expert_key].devices[0]

    def transfer_expert(
        self,
        tensor: torch.Tensor,
        expert_key: ExpertKey,
        target_device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Transfer expert tensor to target device.

        For replicated experts, can target any registered device.
        For sharded, targets the single registered device.

        Args:
            tensor: Expert tensor (typically on CPU).
            expert_key: (layer_id, expert_id) tuple.
            target_device: Device to transfer to. If None, uses primary device.

        Returns:
            Tensor on target_device.

        Raises:
            KeyError: If expert not registered.
            ValueError: If target_device not in expert's registered devices.
        """
        if target_device is None:
            target_device = self.get_primary_device(expert_key)

        allowed_devices = self.expert_placement[expert_key].devices
        if target_device not in allowed_devices:
            raise ValueError(
                f"target_device {target_device} not in allowed devices {allowed_devices}"
            )

        return tensor.to(target_device)

    def distribute_tensor(
        self, tensor: torch.Tensor, expert_key: ExpertKey
    ) -> dict[torch.device, torch.Tensor]:
        """Distribute expert tensor to all registered devices.

        For replicated strategy, sends to all devices.
        For sharded, returns dict with single device.

        Args:
            tensor: Expert tensor (typically on CPU).
            expert_key: (layer_id, expert_id) tuple.

        Returns:
            Mapping from torch.device to tensor on that device.

        Raises:
            KeyError: If expert not registered.
        """
        devices = self.get_devices(expert_key)
        return {device: tensor.to(device) for device in devices}

    def get_expert_placement_summary(self) -> dict[str, object]:
        """Return summary of expert placement across GPUs."""
        return {
            "strategy": self.strategy,
            "num_gpus": self.num_gpus,
            "devices": [str(d) for d in self.devices],
            "registered_experts": len(self.expert_placement),
            "sharded_assignments": self.expert_to_gpu_idx.copy(),
        }

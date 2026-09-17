"""Streaming expert loader for low-resource devices.

Loads experts on-demand from storage, executes, and immediately frees memory.
No caching - predictable memory usage.

IMPORTANT: This module has ZERO dependencies on any cache system.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sparse_llm.inference.expert_processor import ExpertFFN

logger = logging.getLogger(__name__)


class StreamingExpertLoader:
    """Load-execute-free expert loader for streaming mode.

    Key features:
    - No caching - always loads from storage
    - Parallel expert loading for speed
    - Immediate memory release after execution
    - Thread-safe storage access

    This class is completely independent of the cache system.
    """

    def __init__(
        self,
        storage_loader: Callable[[int, int], Dict],
        device: str = "cuda",
        max_parallel: int = 4,
    ):
        """Initialize streaming expert loader.

        Args:
            storage_loader: Function(layer_id, expert_id) -> dict of weights
            device: Target device for computation ("cuda" or "cpu")
            max_parallel: Maximum parallel expert loads (I/O bound)
        """
        self.storage_loader = storage_loader
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.max_parallel = max_parallel
        self.memory_guard = None

    def set_memory_guard(self, guard) -> None:
        """Set memory guard for admission control before GPU transfer."""
        self.memory_guard = guard

    def load_expert(self, layer_id: int, expert_id: int) -> Dict[str, torch.Tensor]:
        """Load expert weights from storage.

        Args:
            layer_id: Layer index
            expert_id: Expert index

        Returns:
            Dict of weight tensors
        """
        return self.storage_loader(layer_id, expert_id)

    def load_experts_parallel(
        self,
        layer_id: int,
        expert_ids: List[int],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Load multiple experts in parallel.

        Args:
            layer_id: Layer index
            expert_ids: List of expert IDs to load

        Returns:
            Dict mapping expert_id -> weights dict
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_id = {
                executor.submit(self.load_expert, layer_id, eid): eid
                for eid in expert_ids
            }

            for future in as_completed(future_to_id):
                eid = future_to_id[future]
                try:
                    results[eid] = future.result()
                except Exception as e:
                    logger.warning(f"Failed to load expert {eid}: {e}")

        return results

    def safe_load_to_device(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Load weights to device with memory check.

        Args:
            weights: Dict of CPU tensors

        Returns:
            Dict of GPU tensors

        Raises:
            MemoryError: If cannot allocate memory
        """
        # Estimate size needed
        size_needed = sum(t.numel() * t.element_size() for t in weights.values())

        # Check memory guard if available
        if self.memory_guard is not None:
            if not self.memory_guard.can_allocate(size_needed):
                # Try to free memory
                if hasattr(self.memory_guard, '_trigger_eviction'):
                    self.memory_guard._trigger_eviction()

                if not self.memory_guard.can_allocate(size_needed):
                    raise MemoryError(f"Cannot allocate {size_needed} bytes for expert weights")

        # Transfer to device
        device_weights = {
            k: v.to(device=self.device) for k, v in weights.items()
        }

        return device_weights

    def load_expert_to_device(self, layer_id: int, expert_id: int) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Load expert and transfer to device with memory guard.

        Args:
            layer_id: Layer index
            expert_id: Expert index

        Returns:
            Tuple of (expert_id, weights on device)
        """
        weights = self.load_expert(layer_id, expert_id)
        device_weights = self.safe_load_to_device(weights)
        return expert_id, device_weights

    def execute_expert(
        self,
        layer_id: int,
        expert_id: int,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Load, execute, and free expert memory.

        This is the core streaming pattern:
        1. Load from storage
        2. Transfer to device
        3. Execute
        4. FREE MEMORY IMMEDIATELY

        Args:
            layer_id: Layer index
            expert_id: Expert index
            hidden_state: Input tensor [batch, hidden]

        Returns:
            Expert output tensor
        """
        # 1. Load from storage
        weights = self.load_expert(layer_id, expert_id)

        # 2. Transfer to device with memory guard
        device_weights = self.safe_load_to_device(weights)

        # 3. Create ExpertFFN and execute
        try:
            w1 = device_weights["w1.weight"]
            w2 = device_weights["w2.weight"]
            expert_dim, hidden_dim = w1.shape
            has_gate = "w3.weight" in device_weights

            expert = ExpertFFN(
                hidden_dim=hidden_dim,
                expert_dim=expert_dim,
                activation="silu",
                has_gate=has_gate,
                dtype=w1.dtype,
            )
            expert = expert.to(self.device)

            # Copy weights
            expert.w1.weight.copy_(w1)
            expert.w2.weight.copy_(w2)
            if has_gate and expert.w3 is not None:
                expert.w3.weight.copy_(device_weights["w3.weight"])
            expert.eval()

            # Execute with no gradients
            with torch.no_grad():
                output = expert(hidden_state.to(self.device))

            return output

        finally:
            # 4. FREE MEMORY IMMEDIATELY
            del expert
            del device_weights
            del weights

            # Clear CUDA cache if on GPU
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def execute_experts_parallel(
        self,
        layer_id: int,
        expert_ids: List[int],
        hidden_state: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Load multiple experts in parallel, execute, and free.

        Args:
            layer_id: Layer index
            expert_ids: List of expert IDs to execute
            hidden_state: Input tensor [batch, hidden]
            expert_weights: Router weights for combining outputs

        Returns:
            Combined expert output tensor
        """
        hidden_dim = hidden_state.shape[-1]

        # Initialize output accumulator
        output = torch.zeros_like(hidden_state)

        # Load experts in parallel
        loaded_experts = self.load_experts_parallel(layer_id, list(expert_ids))

        # Execute each expert
        for expert_id in expert_ids:
            if expert_id not in loaded_experts:
                continue

            device_weights = self.safe_load_to_device(loaded_experts[expert_id])

            try:
                w1 = device_weights["w1.weight"]
                w2 = device_weights["w2.weight"]
                expert_dim, _ = w1.shape
                has_gate = "w3.weight" in device_weights

                expert = ExpertFFN(
                    hidden_dim=hidden_dim,
                    expert_dim=expert_dim,
                    activation="silu",
                    has_gate=has_gate,
                    dtype=w1.dtype,
                ).to(self.device)

                expert.w1.weight.copy_(w1)
                expert.w2.weight.copy_(w2)
                if has_gate and expert.w3 is not None:
                    expert.w3.weight.copy_(device_weights["w3.weight"])
                expert.eval()

                with torch.no_grad():
                    expert_output = expert(hidden_state)

                # Get router weight for this expert
                if expert_weights.dim() == 1:
                    router_w = expert_weights[expert_ids.index(expert_id)]
                else:
                    router_w = expert_weights[expert_ids.index(expert_id)]

                output = output + router_w * expert_output

            finally:
                del expert
                del device_weights
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return output

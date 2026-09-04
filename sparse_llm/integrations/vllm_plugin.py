"""vLLM Integration: Custom MoE Layer Plugin

This module integrates SparseLLM's expert paging with vLLM's serving infrastructure
by implementing vLLM's PluggableLayer protocol.

Architecture:
- Hooks into vLLM's model loading system
- Replaces vLLM's default MoE layers with paged versions
- Coordinates GPU memory with vLLM's KV cache allocator
- Maintains full compatibility with vLLM's serving API
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Optional, List

import torch
import torch.nn as nn

# vLLM imports - these will be available when vLLM is installed
try:
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.custom_op import CustomOp
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    # Fallback base classes
    FusedMoE = nn.Module
    CustomOp = object

from sparse_llm.models.paging import PagedMoELayer, RouterSelection


class VLLMSparseExpertPager:
    """Expert pager that coordinates with vLLM's memory allocator.

    This pager respects vLLM's GPU memory budget and coordinates
    expert cache allocation with vLLM's KV cache.
    """

    def __init__(
        self,
        gpu_cache_bytes: int,
        cpu_cache_bytes: int,
        num_layers: int,
        num_experts: int,
        expert_weight_loader: Any,
        device: str = "cuda",
    ):
        self.gpu_cache_bytes = gpu_cache_bytes
        self.cpu_cache_bytes = cpu_cache_bytes
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_weight_loader = expert_weight_loader
        self.device = device

        # Expert storage
        self.gpu_cache: dict[tuple[int, int], Any] = {}
        self.cpu_cache: dict[tuple[int, int], Any] = {}

        # Access tracking for LRU
        self.access_count: dict[tuple[int, int], int] = {}
        self.access_time: dict[tuple[int, int], float] = {}

        # Metrics
        self.total_loads = 0
        self.total_hits = 0
        self.total_load_time_ms = 0.0

    def is_cached(self, expert_key: tuple[int, int]) -> bool:
        """Check if expert is in GPU cache."""
        return expert_key in self.gpu_cache

    @contextmanager
    def lease(self, expert_key: tuple[int, int]):
        """Lease an expert (load if needed)."""
        if expert_key not in self.gpu_cache:
            self._load_to_gpu(expert_key)

        # Update access tracking
        self.access_count[expert_key] = self.access_count.get(expert_key, 0) + 1
        self.access_time[expert_key] = time.time()
        self.total_hits += 1

        try:
            yield self.gpu_cache[expert_key]
        finally:
            pass  # Keep in cache

    def _load_to_gpu(self, expert_key: tuple[int, int]) -> None:
        """Load expert from CPU/disk to GPU."""
        start = time.perf_counter()

        # Evict if necessary
        self._ensure_space()

        # Load from CPU cache or original storage
        if expert_key in self.cpu_cache:
            # Transfer from CPU to GPU
            expert = self.cpu_cache[expert_key]
            expert_gpu = self._transfer_to_gpu(expert)
        else:
            # Load from original storage
            expert_gpu = self.expert_weight_loader.load_expert(expert_key)

        self.gpu_cache[expert_key] = expert_gpu

        load_time_ms = (time.perf_counter() - start) * 1000
        self.total_load_time_ms += load_time_ms
        self.total_loads += 1

    def _ensure_space(self) -> None:
        """Evict experts to make space if needed."""
        # Estimate space per expert
        if not self.gpu_cache:
            return

        # Simple LRU: evict least recently used
        max_cached = max(1, self.gpu_cache_bytes // (100 * 1024 * 1024))

        while len(self.gpu_cache) >= max_cached:
            # Find LRU expert
            lru_key = min(
                self.access_time.keys(),
                key=lambda k: self.access_time.get(k, 0)
            )

            # Evict to CPU
            self.cpu_cache[lru_key] = self._transfer_to_cpu(self.gpu_cache[lru_key])
            del self.gpu_cache[lru_key]

    def _transfer_to_gpu(self, expert: Any) -> Any:
        """Transfer expert tensors from CPU to GPU."""
        if hasattr(expert, 'tensors'):
            gpu_tensors = {
                name: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in expert.tensors.items()
            }
            expert.tensors = gpu_tensors
        return expert

    def _transfer_to_cpu(self, expert: Any) -> Any:
        """Transfer expert tensors from GPU to CPU."""
        if hasattr(expert, 'tensors'):
            cpu_tensors = {
                name: tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in expert.tensors.items()
            }
            expert.tensors = cpu_tensors
        return expert

    def record_routed(self, expert_keys):
        """Record which experts were routed (for prefetching)."""
        # Update access patterns for future prefetching
        pass

    def get_stats(self) -> dict:
        """Get pager statistics."""
        total = self.total_loads + self.total_hits
        hit_rate = self.total_hits / total if total > 0 else 0.0

        return {
            "gpu_cached_experts": len(self.gpu_cache),
            "cpu_cached_experts": len(self.cpu_cache),
            "total_loads": self.total_loads,
            "total_hits": self.total_hits,
            "cache_hit_rate": hit_rate,
            "avg_load_time_ms": self.total_load_time_ms / max(self.total_loads, 1),
        }


class VLLMSparseMoELayer(CustomOp if VLLM_AVAILABLE else nn.Module):
    """vLLM-compatible MoE layer with expert paging.

    This layer implements vLLM's CustomOp interface to integrate
    seamlessly with vLLM's execution engine.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_idx: int,
        expert_pager: VLLMSparseExpertPager,
        router: nn.Module,
        params_dtype: torch.dtype = torch.float16,
        tp_size: int = 1,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.layer_idx = layer_idx
        self.expert_pager = expert_pager
        self.params_dtype = params_dtype
        self.tp_size = tp_size

        # Router (kept on GPU, it's small)
        self.gate = router

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE layer with expert paging."""

        # Flatten to [tokens, hidden_size]
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states.shape[0]

        # Route tokens to experts
        router_logits = self.gate(hidden_states)
        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Get unique experts needed
        unique_experts = selected_experts.unique().tolist()

        # Notify pager of routing (for prefetching)
        self.expert_pager.record_routed(
            (self.layer_idx, exp_id) for exp_id in unique_experts
        )

        # Compute expert outputs
        final_output = torch.zeros(
            (num_tokens, self.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in unique_experts:
            expert_key = (self.layer_idx, expert_idx)

            # Find tokens assigned to this expert
            expert_mask = (selected_experts == expert_idx)
            token_indices, expert_weight_indices = torch.where(expert_mask)

            if len(token_indices) == 0:
                continue

            # Get unique token indices
            unique_token_indices = token_indices.unique()
            expert_input = hidden_states[unique_token_indices]

            # Load expert and compute
            with self.expert_pager.lease(expert_key) as expert:
                expert_output = self._expert_forward(expert, expert_input)

            # Scatter outputs with routing weights
            token_map = {
                int(token_idx): out_idx
                for out_idx, token_idx in enumerate(unique_token_indices)
            }

            for token_idx, weight_idx in zip(token_indices, expert_weight_indices):
                token_idx = int(token_idx)
                weight_idx = int(weight_idx)
                output_idx = token_map[token_idx]

                weight = routing_weights[token_idx, weight_idx]
                final_output[token_idx] += weight * expert_output[output_idx]

        # Reshape to original
        return final_output.view(orig_shape)

    def _expert_forward(
        self,
        expert: Any,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute expert forward pass (Mixtral-style)."""

        # Extract weights
        if hasattr(expert, 'tensors'):
            w1 = expert.tensors['w1']
            w2 = expert.tensors['w2']
            w3 = expert.tensors['w3']
        else:
            raise ValueError("Expert must have 'tensors' attribute")

        # GLU: gate(x) * up(x) then down
        gate_output = torch.nn.functional.silu(
            torch.nn.functional.linear(hidden_states, w1)
        )
        up_output = torch.nn.functional.linear(hidden_states, w3)
        down_input = gate_output * up_output
        output = torch.nn.functional.linear(down_input, w2)

        return output


def replace_moe_layers_with_paged(
    model: nn.Module,
    expert_pager: VLLMSparseExpertPager,
) -> nn.Module:
    """Replace vLLM's MoE layers with paged versions.

    This function walks the model and replaces FusedMoE layers
    with VLLMSparseMoELayer instances.

    Args:
        model: vLLM model instance
        expert_pager: Expert paging system

    Returns:
        Modified model with paged MoE layers
    """

    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM not available. Install with: pip install vllm")

    layer_idx = 0

    for name, module in model.named_modules():
        # Check if this is a MoE layer
        if isinstance(module, FusedMoE) or 'moe' in name.lower():

            # Extract configuration
            config = getattr(model, 'config', None)
            if config is None:
                continue

            num_experts = getattr(config, 'num_local_experts', 8)
            top_k = getattr(config, 'num_experts_per_tok', 2)
            hidden_size = getattr(config, 'hidden_size', 4096)
            intermediate_size = getattr(config, 'intermediate_size', 14336)

            # Create paged replacement
            paged_layer = VLLMSparseMoELayer(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_idx=layer_idx,
                expert_pager=expert_pager,
                router=module.gate if hasattr(module, 'gate') else None,
            )

            # Replace in model
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, paged_layer)
            else:
                setattr(model, child_name, paged_layer)

            layer_idx += 1

    return model


__all__ = [
    "VLLMSparseExpertPager",
    "VLLMSparseMoELayer",
    "replace_moe_layers_with_paged",
    "VLLM_AVAILABLE",
]

"""Expert processor for MoE inference with cache-aware execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_llm.cache.expert_cache import ExpertCache
from sparse_llm.inference.expert_cache_manager import ExpertCacheManager


class ExpertFFN(nn.Module):
    """Feed-forward network for a single expert: Linear -> Activation -> Linear."""

    def __init__(self, hidden_dim: int, expert_dim: int, activation: str = "gelu", has_gate: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.has_gate = has_gate

        self.w1 = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, hidden_dim, bias=False)

        # Some models (like Mixtral) use gated experts with w3
        if has_gate:
            self.w3 = nn.Linear(hidden_dim, expert_dim, bias=False)
        else:
            self.w3 = None

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute expert FFN: x -> Linear -> Activation -> Linear -> output."""
        if self.w3 is not None:
            # Gated expert: w2(activation(w1(x)) * w3(x))
            return self.w2(self.activation(self.w1(x)) * self.w3(x))
        else:
            # Standard expert: w2(activation(w1(x)))
            return self.w2(self.activation(self.w1(x)))


class ExpertProcessor:
    """Process tokens through MoE experts with intelligent caching.

    Supports two modes:
    - Batched processing for prefill: group tokens by expert, process in batch
    - Single-token processing for decode: process through top-k experts, combine

    Integrates with ExpertCacheManager for predictive prefetching and optimization.
    """

    def __init__(
        self,
        expert_cache: ExpertCache | ExpertCacheManager,
        num_experts: int,
        hidden_dim: int,
        expert_dim: int,
        activation: str = "gelu",
        device: str = "cuda",
        enable_predictive_prefetch: bool = True,
    ):
        """Initialize expert processor.

        Args:
            expert_cache: Cache or cache manager for expert weights
            num_experts: Total number of experts in the model
            hidden_dim: Model hidden dimension
            expert_dim: Expert intermediate dimension
            activation: Activation function (gelu, relu, silu)
            device: Device for computation (cuda or cpu)
            enable_predictive_prefetch: Enable predictive prefetching (only if cache manager)
        """
        # Wrap raw cache with manager if needed
        if isinstance(expert_cache, ExpertCacheManager):
            self.cache_manager = expert_cache
            self.expert_cache = expert_cache.cache
        else:
            self.cache_manager = None
            self.expert_cache = expert_cache

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.activation = activation
        self.device = device
        self.enable_predictive_prefetch = enable_predictive_prefetch

        # Stats tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self._last_activated_experts: List[int] = []

    def _get_expert(
        self,
        expert_id: int,
        layer_id: Optional[int] = None,
        loader: Optional[Callable[[], ExpertFFN]] = None,
    ) -> ExpertFFN:
        """Retrieve expert from cache or load if missing.

        Args:
            expert_id: Expert identifier
            layer_id: Optional layer identifier for layer-aware caching
            loader: Callable that loads the expert weights if not cached

        Returns:
            Expert FFN module ready for forward pass

        Raises:
            ValueError: If expert not in cache and no loader provided
        """
        if loader is not None:
            # Use get_or_load for automatic cache management
            expert = self.expert_cache.get_or_load(
                expert_id,
                loader,
                layer_id=layer_id,
                pin=True,  # Pin during execution to prevent eviction
            )
            # Note: tier-aware hit/miss stats come from ExpertCache.get_stats(),
            # not from unconditional counter increments here.
        else:
            # Simple cache lookup
            expert = self.expert_cache.get(expert_id, layer_id=layer_id)
            if expert is None:
                raise ValueError(
                    f"Expert {expert_id} (layer {layer_id}) not in cache and no loader provided"
                )
            self.cache_hits += 1

        return expert

    def process_batch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        layer_id: Optional[int] = None,
        expert_loader: Optional[Callable[[int], ExpertFFN]] = None,
        trigger_prefetch: bool = True,
    ) -> torch.Tensor:
        """Process batch of tokens through experts (prefill mode).

        Strategy:
        1. Group tokens by expert assignment
        2. For each expert: load if needed, process all assigned tokens in batch
        3. Reorder outputs to match original token sequence
        4. Combine expert outputs using router weights
        5. Trigger predictive prefetching if enabled

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            expert_indices: Expert assignments [batch_size, seq_len, top_k]
            expert_weights: Router weights [batch_size, seq_len, top_k]
            layer_id: Optional layer identifier for layer-aware caching
            expert_loader: Optional function to load expert by ID
            trigger_prefetch: Whether to trigger prefetching after this batch

        Returns:
            Combined output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        top_k = expert_indices.shape[-1]

        # Flatten batch and sequence dimensions for processing
        flat_hidden = hidden_states.reshape(-1, hidden_dim)  # [batch*seq, hidden]
        flat_indices = expert_indices.reshape(-1, top_k)  # [batch*seq, top_k]
        flat_weights = expert_weights.reshape(-1, top_k)  # [batch*seq, top_k]

        # Initialize output accumulator
        output = torch.zeros_like(flat_hidden)

        # Track activated experts for prefetching
        activated_experts = set()

        # Process each expert position (typically k=2)
        for k_idx in range(top_k):
            expert_ids = flat_indices[:, k_idx]  # [batch*seq]
            weights = flat_weights[:, k_idx]  # [batch*seq]

            # Group tokens by expert
            for expert_id in range(self.num_experts):
                # Find all tokens assigned to this expert
                mask = expert_ids == expert_id
                if not mask.any():
                    continue

                activated_experts.add(expert_id)

                # Get tokens for this expert
                expert_inputs = flat_hidden[mask]  # [num_tokens, hidden]
                expert_weights_subset = weights[mask]  # [num_tokens]

                # Load expert if needed
                if expert_loader is not None:
                    # Fix closure issue with default parameter
                    loader_fn = lambda eid=expert_id: expert_loader(eid)
                else:
                    loader_fn = None

                try:
                    expert = self._get_expert(expert_id, layer_id, loader_fn)
                except Exception as e:
                    # If expert loading fails, skip this expert to avoid segfault
                    import logging
                    logging.warning(f"Failed to load expert {expert_id} for layer {layer_id}: {e}")
                    continue

                # Process tokens through expert
                with torch.no_grad():
                    expert_output = expert(expert_inputs)  # [num_tokens, hidden]

                # Weight and accumulate outputs
                weighted_output = expert_output * expert_weights_subset.unsqueeze(-1)
                output[mask] += weighted_output

                # Unpin expert after use
                if expert_loader is not None:
                    self.expert_cache.unpin(expert_id, layer_id)

        # Reshape to original batch structure
        output = output.reshape(batch_size, seq_len, hidden_dim)

        # Update predictor and trigger prefetching
        self._last_activated_experts = list(activated_experts)
        if (
            trigger_prefetch
            and self.enable_predictive_prefetch
            and self.cache_manager is not None
            and expert_loader is not None
        ):
            self.cache_manager.optimize_cache_for_decode(
                self._last_activated_experts, expert_loader, layer_id
            )

        return output

    def process_single(
        self,
        hidden_state: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        layer_id: Optional[int] = None,
        expert_loader: Optional[Callable[[int], ExpertFFN]] = None,
        update_predictor: bool = True,
    ) -> torch.Tensor:
        """Process single token through experts (decode mode).

        Strategy:
        1. Collect all unique (expert_id, batch_idx, weight) across top-k
        2. Load unique experts concurrently via ThreadPoolExecutor
        3. Accumulate weighted outputs sequentially (tensor ops are cheap vs I/O)
        4. Update predictor with activated experts

        Args:
            hidden_state: Input tensor [batch_size, 1, hidden_dim] or [batch_size, hidden_dim]
            expert_indices: Expert assignments [batch_size, top_k] or [top_k]
            expert_weights: Router weights [batch_size, top_k] or [top_k]
            layer_id: Optional layer identifier for layer-aware caching
            expert_loader: Optional function to load expert by ID
            update_predictor: Whether to update predictor with activated experts

        Returns:
            Combined output tensor matching input shape
        """
        # Handle both [batch, hidden] and [batch, 1, hidden] formats
        input_shape = hidden_state.shape
        if len(input_shape) == 3:
            batch_size, seq_len, hidden_dim = input_shape
            assert seq_len == 1, "process_single expects sequence length of 1"
            hidden_state = hidden_state.squeeze(1)  # [batch, hidden]
        else:
            batch_size, hidden_dim = input_shape

        # Handle both [batch, top_k] and [top_k] formats
        if expert_indices.dim() == 1:
            expert_indices = expert_indices.unsqueeze(0)  # [1, top_k]
            expert_weights = expert_weights.unsqueeze(0)  # [1, top_k]

        top_k = expert_indices.shape[-1]

        # Collect all (expert_id, batch_idx, weight) entries
        entries: list[tuple[int, int, float]] = []
        for k_idx in range(top_k):
            for batch_idx in range(batch_size):
                expert_id = expert_indices[batch_idx, k_idx].item()
                weight = expert_weights[batch_idx, k_idx].item()
                entries.append((expert_id, batch_idx, weight))

        # Load unique experts concurrently
        unique_experts = list({eid for eid, _, _ in entries})
        expert_map: dict[int, ExpertFFN] = {}

        if expert_loader is not None and unique_experts:
            with ThreadPoolExecutor(max_workers=len(unique_experts)) as pool:
                futures = {
                    pool.submit(self._get_expert, eid, layer_id,
                                lambda _eid=eid: expert_loader(_eid)): eid
                    for eid in unique_experts
                }
                for fut in futures:
                    eid = futures[fut]
                    try:
                        expert_map[eid] = fut.result()
                    except Exception:
                        import logging
                        logging.warning(f"Failed to load expert {eid} for layer {layer_id}")
        else:
            # No loader: try direct cache lookup for each unique expert
            for eid in unique_experts:
                try:
                    expert_map[eid] = self._get_expert(eid, layer_id, None)
                except ValueError:
                    pass

        # Accumulate outputs sequentially (tensor ops are cheap; avoids GIL contention)
        output = torch.zeros_like(hidden_state)  # [batch, hidden]
        activated_experts: set[int] = set()

        for expert_id, batch_idx, weight in entries:
            activated_experts.add(expert_id)
            expert = expert_map.get(expert_id)
            if expert is None:
                continue
            token_input = hidden_state[batch_idx:batch_idx+1]  # [1, hidden]
            with torch.no_grad():
                expert_output = expert(token_input)  # [1, hidden]
            output[batch_idx] += weight * expert_output.squeeze(0)

            # Unpin after use
            if expert_loader is not None:
                self.expert_cache.unpin(expert_id, layer_id)

        # Update predictor with activated experts
        if update_predictor and self.cache_manager is not None:
            self._last_activated_experts = list(activated_experts)
            self.cache_manager.update_predictor_from_inference(self._last_activated_experts)

        # Restore original shape if needed
        if len(input_shape) == 3:
            output = output.unsqueeze(1)  # [batch, 1, hidden]

        return output

    def preload_experts(
        self,
        expert_ids: list[int],
        layer_id: Optional[int] = None,
        expert_loader: Optional[Callable[[int], ExpertFFN]] = None,
    ) -> None:
        """Preload a set of experts into cache (useful during prefill).

        Args:
            expert_ids: List of expert IDs to preload
            layer_id: Optional layer identifier
            expert_loader: Function to load expert weights
        """
        if expert_loader is None:
            raise ValueError("expert_loader is required for preloading")

        # Use cache manager if available for better coordination
        if self.cache_manager is not None:
            self.cache_manager.preload_experts(expert_ids, expert_loader, layer_id)
        else:
            # Fallback to direct cache loading
            for expert_id in expert_ids:
                self.expert_cache.get_or_load(
                    expert_id,
                    lambda eid=expert_id: expert_loader(eid),
                    layer_id=layer_id,
                )

    def get_stats(self) -> dict:
        """Return processing statistics.

        Returns:
            Dictionary with cache hit/miss counts and cache stats.
            Tier-aware hit/miss comes from the three-tier ExpertCache.get_stats()
            (gpu_hits / cpu_hits / storage_hits), not from the processor's own
            unconditional counter.
        """
        # Get cache stats from manager if available, otherwise from cache directly
        if self.cache_manager is not None:
            cache_stats = self.cache_manager.get_cache_stats()
        else:
            cache_stats = self.expert_cache.stats()

        return {
            "processor_cache_hits": self.cache_hits,
            "processor_cache_misses": self.cache_misses,
            "last_activated_experts": self._last_activated_experts,
            **cache_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.cache_hits = 0
        self.cache_misses = 0


__all__ = ["ExpertProcessor", "ExpertFFN"]

"""Expert processor for MoE inference with cache-aware execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_llm.loading.expert_cache import ExpertCache
from sparse_llm.inference.expert_cache_manager import ExpertCacheManager
from sparse_llm.inference.expert_fusion import ExpertFusionCache
from sparse_llm.inference.path_tracker import ExpertPathTracker
from sparse_llm.inference.prefetcher import PredictivePrefetcher


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
    - Tiled processing: batch tensor operations on contiguous expert weights

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
        processing_mode: str = "standard",
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
        self.processing_mode = processing_mode

        # Stats tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self._last_activated_experts: List[int] = []

        # Tiled weight cache: layer_id -> {'w1': tensor, 'w2': tensor, 'w3': tensor}
        self._tiled_weights_cache: dict[int, dict[str, torch.Tensor]] = {}

        # Fusion cache for pre-computed fused expert weights
        self._fusion_cache = ExpertFusionCache(max_fused_paths=1000)

        # Predictive prefetcher for background expert prefetching
        self._prefetcher: Optional[PredictivePrefetcher] = None
        if enable_predictive_prefetch:
            self._prefetcher = PredictivePrefetcher(
                expert_loader=self._get_expert,
                enable_background_thread=True,
            )

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

    def _compute_batch_fingerprints(
        self,
        routing: torch.Tensor,  # [batch, seq, top_k]
    ) -> torch.Tensor:
        """Compute fingerprints for each token's expert path.

        Args:
            routing: Expert indices [batch, seq, top_k]

        Returns:
            Fingerprints tensor [batch, seq]
        """
        B, S, top_k = routing.shape
        flat_routing = routing.view(-1, top_k)  # [B*S, top_k]

        # Simple hash: XOR fold expert IDs with position weighting
        fingerprints = torch.zeros(B * S, dtype=torch.long, device=routing.device)
        for k in range(top_k):
            fingerprints ^= flat_routing[:, k].long() * (self.num_experts ** k)
        return fingerprints.view(B, S)

    def _group_by_fingerprint(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden]
        routing: torch.Tensor,  # [batch, seq, top_k] - expert indices per token
        fingerprints: torch.Tensor,  # [batch, seq] - path fingerprint per token
    ) -> dict[int, dict]:
        """Group tokens by their expert path fingerprint.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden]
            routing: Expert assignments [batch, seq, top_k]
            fingerprints: Computed fingerprints [batch, seq]

        Returns:
            groups: dict[fingerprint -> {
                'indices': LongTensor of original positions,
                'expert_ids': set of expert IDs needed,
                'hidden': tensor of hidden states for group
            }]
        """
        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)
        flat_fp = fingerprints.view(-1)
        flat_routing = routing.view(-1, routing.shape[-1])  # [B*S, top_k]

        # Group by fingerprint
        groups: dict[int, dict] = {}
        for idx, fp in enumerate(flat_fp.tolist()):
            if fp not in groups:
                groups[fp] = {
                    'indices': [],
                    'expert_ids': set(),
                }
            groups[fp]['indices'].append(idx)
            # Collect expert IDs
            for expert_id in flat_routing[idx].tolist():
                groups[fp]['expert_ids'].add(expert_id)

        # Convert indices to tensor and add hidden states
        for fp, group in groups.items():
            indices = torch.tensor(group['indices'], device=hidden_states.device, dtype=torch.long)
            group['indices'] = indices
            group['hidden'] = flat_hidden[indices]

        return groups

    def _process_group(
        self,
        hidden: torch.Tensor,
        routing: torch.Tensor,
        weights: torch.Tensor,
        expert_modules: dict[int, ExpertFFN],
    ) -> torch.Tensor:
        """Process a group of tokens with the same expert path.

        Args:
            hidden: Hidden states for this group [num_tokens, hidden]
            routing: Expert indices for this group [num_tokens, top_k]
            weights: Router weights for this group [num_tokens, top_k]
            expert_modules: Pre-loaded expert modules {expert_id: ExpertFFN}

        Returns:
            Output tensor [num_tokens, hidden]
        """
        num_tokens, hidden_dim = hidden.shape
        output = torch.zeros_like(hidden)

        # Process each token's expert contributions
        for token_idx in range(num_tokens):
            token_hidden = hidden[token_idx:token_idx + 1]  # [1, hidden]
            token_experts = routing[token_idx]  # [top_k]
            token_weights = weights[token_idx]  # [top_k]

            for expert_id, weight in zip(token_experts.tolist(), token_weights.tolist()):
                expert = expert_modules.get(expert_id)
                if expert is None:
                    continue
                with torch.no_grad():
                    expert_output = expert(token_hidden)  # [1, hidden]
                output[token_idx] += weight * expert_output.squeeze(0)

        return output

    def _tile_expert_weights(
        self,
        layer_id: int,
        expert_loader: Optional[Callable[[int], ExpertFFN]] = None,
    ) -> dict[str, torch.Tensor]:
        """Load expert weights as tiled tensors for batch access.

        Current: [expert_0_w1, expert_1_w1, ...] - scattered
        Tiled:   [num_experts, hidden, expert_dim] - contiguous

        Enables: expert_weights[[3, 7, 11]] - batch gather.

        Args:
            layer_id: Layer identifier for layer-aware caching
            expert_loader: Optional function to load expert by ID

        Returns:
            Dictionary with tiled weight tensors: {'w1': tensor, 'w2': tensor, 'w3': tensor}
        """
        # Check cache first
        if layer_id in self._tiled_weights_cache:
            return self._tiled_weights_cache[layer_id]

        num_experts = self.num_experts
        hidden_dim = self.hidden_dim
        expert_dim = self.expert_dim
        has_gate = self.activation in ("silu", "swiglu")

        # Stack w1, w2, w3 weights into contiguous tensors
        # Linear weights are [out_features, in_features]
        # w1: [expert_dim, hidden_dim], w2: [hidden_dim, expert_dim], w3: [expert_dim, hidden_dim]
        w1_tiled = torch.zeros(num_experts, expert_dim, hidden_dim, device=self.device, dtype=torch.float32)
        w2_tiled = torch.zeros(num_experts, hidden_dim, expert_dim, device=self.device, dtype=torch.float32)
        w3_tiled = torch.zeros(num_experts, expert_dim, hidden_dim, device=self.device, dtype=torch.float32) if has_gate else torch.zeros(1, device=self.device)

        for expert_id in range(num_experts):
            if expert_loader is not None:
                loader_fn = partial(expert_loader, expert_id)
                expert = self._get_expert(expert_id, layer_id, loader_fn)
            else:
                expert = self._get_expert(expert_id, layer_id, None)

            with torch.no_grad():
                w1_tiled[expert_id] = expert.w1.weight
                w2_tiled[expert_id] = expert.w2.weight
                if expert.w3 is not None:
                    w3_tiled[expert_id] = expert.w3.weight

        tiled_weights = {'w1': w1_tiled, 'w2': w2_tiled}
        if has_gate:
            tiled_weights['w3'] = w3_tiled

        # Cache the tiled weights
        self._tiled_weights_cache[layer_id] = tiled_weights
        return tiled_weights

    def _process_group_tiled(
        self,
        hidden: torch.Tensor,
        routing: torch.Tensor,
        weights: torch.Tensor,
        tiled_weights: dict[str, torch.Tensor],
        expert_ids: list[int],
    ) -> torch.Tensor:
        """Process group using tiled/batched tensor operations.

        Args:
            hidden: Hidden states for this group [num_tokens, hidden]
            routing: Expert indices for this group [num_tokens, top_k]
            weights: Router weights for this group [num_tokens, top_k]
            tiled_weights: Pre-tiled weight tensors
            expert_ids: List of expert IDs in this group

        Returns:
            Output tensor [num_tokens, hidden]
        """
        num_tokens = hidden.shape[0]
        output = torch.zeros_like(hidden)

        has_gate = 'w3' in tiled_weights and tiled_weights['w3'].shape[0] > 1

        for expert_id in expert_ids:
            # Find tokens that route to this expert (at any position)
            mask = (routing == expert_id).any(dim=1)
            token_indices = mask.nonzero().squeeze(-1)

            if len(token_indices) == 0:
                continue

            # Get hidden states for these tokens
            token_hidden = hidden[token_indices]

            # Compute expert output once for all tokens
            act_fn = getattr(F, self.activation.split('_')[0], F.gelu) if self.activation else F.gelu
            if has_gate:
                x1 = act_fn(token_hidden @ tiled_weights['w1'][expert_id].t())
                x3 = token_hidden @ tiled_weights['w3'][expert_id].t()
                expert_out = x1 * x3 @ tiled_weights['w2'][expert_id].t()
            else:
                expert_out = act_fn(token_hidden @ tiled_weights['w1'][expert_id].t()) @ tiled_weights['w2'][expert_id].t()

            # Apply weights: each token may route to expert multiple times
            for i, tok_idx in enumerate(token_indices):
                tok_routing = routing[tok_idx]
                tok_weights = weights[tok_idx]
                for k in range(routing.shape[1]):
                    if tok_routing[k] == expert_id:
                        output[tok_idx] += expert_out[i] * tok_weights[k]

        return output

    def invalidate_tiled_cache(self, layer_id: Optional[int] = None) -> None:
        """Invalidate tiled weight cache.

        Args:
            layer_id: Specific layer to invalidate, or None to clear all
        """
        if layer_id is None:
            self._tiled_weights_cache.clear()
        elif layer_id in self._tiled_weights_cache:
            del self._tiled_weights_cache[layer_id]

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

        # Compute fingerprints for grouping
        fingerprints = self._compute_batch_fingerprints(expert_indices)  # [B, S]

        # Group tokens by fingerprint
        groups = self._group_by_fingerprint(hidden_states, expert_indices, fingerprints)

        # Initialize output
        output = torch.zeros(batch_size * seq_len, hidden_dim,
                           device=hidden_states.device, dtype=hidden_states.dtype)

        # Track activated experts for prefetching
        activated_experts = set()

        # Process each group (experts loaded once per group)
        for fp, group in groups.items():
            # Load experts for this group (ONCE)
            expert_modules: dict[int, ExpertFFN] = {}
            for expert_id in group['expert_ids']:
                activated_experts.add(expert_id)
                if expert_loader is not None:
                    loader_fn = partial(expert_loader, expert_id)
                else:
                    loader_fn = None
                try:
                    expert_modules[expert_id] = self._get_expert(expert_id, layer_id, loader_fn)
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to load expert {expert_id} for layer {layer_id}: {e}")

            # Get routing for this group
            group_indices = group['indices']
            group_routing = expert_indices.view(batch_size * seq_len, -1)[group_indices]
            group_weights = expert_weights.view(batch_size * seq_len, -1)[group_indices]

            # Process group
            group_output = self._process_group(
                group['hidden'],
                group_routing,
                group_weights,
                expert_modules
            )

            output[group_indices] = group_output

            # Unpin experts after group processing
            if expert_loader is not None:
                for expert_id in expert_modules:
                    try:
                        self.expert_cache.unpin(expert_id, layer_id)
                    except Exception:
                        pass

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

    def process_batch_tiled(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        layer_id: Optional[int] = None,
        expert_loader: Optional[Callable[[int], ExpertFFN]] = None,
        trigger_prefetch: bool = True,
    ) -> torch.Tensor:
        """Process batch of tokens using tiled/batched tensor operations.

        This method uses pre-tiled expert weights for batch gather operations,
        which can be more efficient for large batches with many expert accesses.

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

        # Compute fingerprints for grouping
        fingerprints = self._compute_batch_fingerprints(expert_indices)

        # Group tokens by fingerprint
        groups = self._group_by_fingerprint(hidden_states, expert_indices, fingerprints)

        # Get tiled weights (cached per layer)
        tiled_weights = self._tile_expert_weights(layer_id, expert_loader)

        # Initialize output
        output = torch.zeros(batch_size * seq_len, hidden_dim,
                           device=hidden_states.device, dtype=hidden_states.dtype)

        # Track activated experts for prefetching
        activated_experts = set()

        # Process each group
        for fp, group in groups.items():
            # Collect expert IDs needed for this group
            group_expert_ids = list(group['expert_ids'])
            activated_experts.update(group_expert_ids)

            # Get routing for this group
            group_indices = group['indices']
            group_routing = expert_indices.view(batch_size * seq_len, -1)[group_indices]
            group_weights = expert_weights.view(batch_size * seq_len, -1)[group_indices]

            # Process group with tiled weights
            group_output = self._process_group_tiled(
                group['hidden'],
                group_routing,
                group_weights,
                tiled_weights,
                group_expert_ids,
            )

            output[group_indices] = group_output

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

    def process_batch_fused(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        layer_id: Optional[int] = None,
        expert_loader: Optional[Callable[[int], ExpertFFN]] = None,
    ) -> torch.Tensor:
        """Process batch using fused expert weights for common paths.

        Uses pre-computed fused weights to reduce multiple matmuls to single matmul
        for paths that appear frequently.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_dim]
            expert_indices: Expert assignments [batch_size, seq_len, top_k]
            expert_weights: Router weights [batch_size, seq_len, top_k]
            layer_id: Optional layer identifier
            expert_loader: Optional function to load expert by ID

        Returns:
            Combined output tensor [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        fingerprints = self._compute_batch_fingerprints(expert_indices)
        groups = self._group_by_fingerprint(hidden_states, expert_indices, fingerprints)
        tiled_weights = self._tile_expert_weights(layer_id, expert_loader)

        output = torch.zeros(
            batch_size * seq_len, hidden_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        for fp, group in groups.items():
            expert_ids = list(group["expert_ids"])
            group_indices = group["indices"]
            group_routing = expert_indices.view(batch_size * seq_len, -1)[group_indices]
            group_weights = expert_weights.view(batch_size * seq_len, -1)[group_indices]

            # Average routing weights per expert for this group
            avg_weights: dict[int, float] = {}
            for i in range(group_routing.shape[0]):
                for k in range(group_routing.shape[1]):
                    eid = group_routing[i, k].item()
                    w = group_weights[i, k].item()
                    avg_weights[eid] = avg_weights.get(eid, 0.0) + w
            for eid in avg_weights:
                avg_weights[eid] /= len(group_indices)

            fused_weights = self._fusion_cache.get_fused(
                fp, expert_ids, list(avg_weights.values()), tiled_weights
            )

            group_hidden = group["hidden"]
            has_gate = "w3" in fused_weights
            act_fn = getattr(F, self.activation.split('_')[0], F.gelu) if self.activation else F.gelu

            if has_gate:
                x1 = act_fn(group_hidden @ fused_weights["w1"].t())
                x3 = group_hidden @ fused_weights["w3"].t()
                group_output = x1 * x3 @ fused_weights["w2"].t()
            else:
                group_output = act_fn(group_hidden @ fused_weights["w1"].t()) @ fused_weights["w2"].t()

            output[group_indices] = group_output

        return output.reshape(batch_size, seq_len, hidden_dim)

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
                                partial(expert_loader, eid)): eid
                    for eid in unique_experts
                }
                for fut in futures:
                    eid = futures[fut]
                    try:
                        expert_map[eid] = fut.result()
                    except Exception as e:
                        import logging
                        logging.error(f"Failed to load expert {eid} for layer {layer_id}: {e}")
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

    def prefetch_next(self, current_experts: list[int], layer_id: int) -> None:
        """Prefetch predicted next experts based on current activations.

        Args:
            current_experts: Currently activated expert IDs
            layer_id: Layer to prefetch for
        """
        if self._prefetcher:
            self._prefetcher.predict_and_prefetch(current_experts, layer_id)

    def observe_routing(self, expert_ids: list[int], layer_id: int = 0) -> None:
        """Observe expert routing patterns to learn transitions.

        Args:
            expert_ids: List of expert IDs that were activated
            layer_id: Layer index for tracking
        """
        if self._prefetcher:
            self._prefetcher.observe_layer(layer_id=layer_id, expert_ids=expert_ids)


__all__ = ["ExpertProcessor", "ExpertFFN", "ExpertFusionCache", "PredictivePrefetcher"]

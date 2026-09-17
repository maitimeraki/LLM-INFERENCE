"""
Attention Engine for MoE Inference

Handles both prefill (full self-attention) and decode (incremental with KV cache) modes.
Optimized for low-latency token generation with efficient cache management.

Layer-loop ownership (Ruling R7): ``prefill``/``decode`` process exactly ONE layer and
take an explicit ``layer_idx``. The per-layer loop lives in ``CustomMoEInferenceEngine``.
The KV cache buffer stays shaped ``[batch, num_layers, max_seq_len, kv_heads, head_dim]``
and a single-layer call reads/writes only ``[:, layer_idx, ...]``.

seq_len contract (R7): ``decode`` writes K/V at ``current_pos = kv_cache['seq_len']`` and
does NOT advance it. The engine advances ``kv_cache['seq_len']`` exactly once per
generated token, after its full layer loop.
"""

import logging
import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Weight-only RMSNorm: ``x * rsqrt(mean(x^2) + eps) * weight``.

    eps defaults to 1e-6. The mean is accumulated in fp32 and the result cast
    back to the input dtype to match the reference model's numerics.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


class AttentionEngine:
    """
    Attention engine supporting prefill and decode phases with KV caching.

    Performance targets:
    - Prefill: 1-3s for 1k tokens (full self-attention)
    - Decode: 20-30ms per token (incremental with cache)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        num_kv_heads: Optional[int] = None,
    ):
        """
        Initialize attention engine.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            max_seq_len: Maximum sequence length supported
            device: Device to run on ("cuda" or "cpu")
            dtype: Data type for computations (fp16 for speed, fp32 for stability)
            num_kv_heads: Key/value head count for GQA/MQA. Defaults to num_heads
                (standard MHA), so existing callers are unaffected.
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(head_dim)

        # Check if FlashAttention is available (torch >= 2.0)
        self.use_flash_attention = hasattr(F, 'scaled_dot_product_attention')

        # Per-layer learned projections, populated by load_layer_weights().
        # Empty dict => legacy reshape-only attention with a warning.
        self.layer_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self._warned_proj = set()

        # OPTIMIZATION: Pre-allocate KV cache buffers for max sequence length
        # This eliminates the need to recreate and copy tensors on every decode step
        # Memory cost: 2 * num_layers * max_seq_len * num_heads * head_dim * bytes_per_element
        # For 32 layers, 32 heads, 128 head_dim, 4096 tokens, fp16: ~2GB
        # This is a one-time allocation that persists across generation steps
        self._kv_cache_buffer = None
        self._cache_initialized = False

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_layer_weights(self, shared_weights: dict, layer_id: int) -> None:
        """Pull one layer's Q/K/V/O projection weights out of a checkpoint dict.

        Accepts both ``model.layers.{L}.self_attn.*`` and ``layers.{L}.self_attn.*``
        key prefixes. Missing projections are recorded here (warning once per
        projection) so the attention path can fall back to the legacy reshape.
        """
        prefixes = (
            f"model.layers.{layer_id}.self_attn.",
            f"layers.{layer_id}.self_attn.",
        )
        layer_w: Dict[str, torch.Tensor] = {}
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            for prefix in prefixes:
                key = f"{prefix}{name}.weight"
                if key in shared_weights:
                    layer_w[name] = shared_weights[key]
                    break
            else:
                self._warn_once(
                    name,
                    f"Layer {layer_id}: missing self_attn.{name}.weight; "
                    f"using identity fallback for this projection.",
                )
        self.layer_weights[layer_id] = layer_w

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned_proj:
            self._warned_proj.add(key)
            logger.warning(message)

    # ------------------------------------------------------------------
    # Prefill / decode (exactly one layer each)
    # ------------------------------------------------------------------
    def prefill(
        self,
        input_ids: torch.Tensor,
        layer_idx: int = 0,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Prefill phase: process the whole prompt through ONE layer.

        Args:
            input_ids: Input token embeddings [batch, seq_len, hidden_dim]
            layer_idx: Which layer's projections/cache slot to use
            kv_cache: Optional pre-allocated cache. Created if None.
            attention_mask: Optional mask [batch, seq_len] (1 valid, 0 pad)

        Returns:
            hidden_states: Output embeddings [batch, seq_len, hidden_dim]
            kv_cache: The (possibly newly created) cache dict.
        """
        batch_size, seq_len, hidden_dim = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Hidden dim {hidden_dim} does not match expected {self.hidden_dim}")

        # Move to device and cast to dtype
        input_ids = input_ids.to(self.device, dtype=self.dtype)

        # Create the pre-allocated KV cache buffer on first use
        if kv_cache is None:
            kv_cache = {
                'keys': torch.zeros(
                    batch_size, self.num_layers, self.max_seq_len, self.num_kv_heads, self.head_dim,
                    device=self.device, dtype=self.dtype
                ),
                'values': torch.zeros(
                    batch_size, self.num_layers, self.max_seq_len, self.num_kv_heads, self.head_dim,
                    device=self.device, dtype=self.dtype
                ),
                'seq_len': 0,
            }

        # Process attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            # Convert to additive mask: 0 for valid positions, -inf for padding
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

        # Learned Q/K/V projections for this layer
        q, k, v = self._project_qkv(input_ids, layer_idx)

        # Store K, V in this layer's cache slot (no cross-layer writes)
        kv_cache['keys'][:, layer_idx, :seq_len] = k.transpose(1, 2)
        kv_cache['values'][:, layer_idx, :seq_len] = v.transpose(1, 2)

        # Repeat KV heads for GQA/MQA so shapes match Q
        q_k, k_full, v_full = self._for_attention(q, k, v)

        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                q_k, k_full, v_full,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=(attention_mask is None)
            )
        else:
            attn_output = self._manual_attention(q_k, k_full, v_full, attention_mask)

        # Reshape back to [batch, seq_len, hidden_dim] and apply output projection
        hidden_states = self._reshape_from_attention(attn_output)
        hidden_states = self._apply_out_proj(hidden_states, layer_idx)

        # Prefill writes a full prompt; seq_len is the prompt length and is
        # idempotent across the engine's per-layer calls.
        kv_cache['seq_len'] = seq_len

        return hidden_states, kv_cache

    def decode(
        self,
        token_id: torch.Tensor,
        layer_idx: int,
        kv_cache: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode phase: process a single token through ONE layer using cached K/V.

        Args:
            token_id: New token embedding [batch, 1, hidden_dim]
            layer_idx: Which layer's projections/cache slot to use
            kv_cache: Cache holding 'keys', 'values', 'seq_len'

        Returns:
            hidden_states: Output embedding for new token [batch, 1, hidden_dim]
            kv_cache: Same buffer reference (seq_len NOT advanced here).
        """
        batch_size, seq_len, hidden_dim = token_id.shape

        if seq_len != 1:
            raise ValueError(f"Decode expects single token, got seq_len={seq_len}")

        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Hidden dim {hidden_dim} does not match expected {self.hidden_dim}")

        # Move to device and cast to dtype
        token_id = token_id.to(self.device, dtype=self.dtype)

        # Get current position in cache
        current_pos = kv_cache['seq_len']
        next_pos = current_pos + 1

        if next_pos > self.max_seq_len:
            raise ValueError(f"Sequence length {next_pos} exceeds maximum {self.max_seq_len}")

        # Learned Q/K/V projections for this layer
        q, k, v = self._project_qkv(token_id, layer_idx)

        # Write new K, V directly into this layer's slot at current_pos.
        kv_cache['keys'][:, layer_idx, current_pos:next_pos, :, :] = k.transpose(1, 2)
        kv_cache['values'][:, layer_idx, current_pos:next_pos, :, :] = v.transpose(1, 2)

        # Views into the cache (no copy)
        past_k = kv_cache['keys'][:, layer_idx, :next_pos, :, :]
        past_v = kv_cache['values'][:, layer_idx, :next_pos, :, :]
        full_k = past_k.transpose(1, 2)
        full_v = past_v.transpose(1, 2)

        q_k, k_full, v_full = self._for_attention(q, full_k, full_v)

        if self.use_flash_attention:
            attn_output = F.scaled_dot_product_attention(
                q_k, k_full, v_full,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
        else:
            attn_output = self._manual_attention(q_k, k_full, v_full, mask=None)

        # Reshape back to [batch, 1, hidden_dim] and apply output projection
        hidden_states = self._reshape_from_attention(attn_output)
        hidden_states = self._apply_out_proj(hidden_states, layer_idx)

        # R7 seq_len contract: decode writes K/V at current_pos and does NOT
        # advance kv_cache['seq_len']. The engine advances it exactly once per
        # generated token, after its full layer loop. Advancing here would
        # overshoot by num_layers per token.
        return hidden_states, kv_cache

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    def _project_qkv(self, x: torch.Tensor, layer_idx: int):
        """Return Q/K/V as [batch, heads, seq, head_dim] using learned weights.

        Falls back to the legacy reshape-only path (warning once) when the
        checkpoint lacks the projection keys.
        """
        lw = self.layer_weights.get(layer_idx, {})
        if "q_proj" in lw and "k_proj" in lw and "v_proj" in lw:
            q = self._reshape_heads(x @ lw["q_proj"].to(x.dtype).t(), self.num_heads)
            k = self._reshape_heads(x @ lw["k_proj"].to(x.dtype).t(), self.num_kv_heads)
            v = self._reshape_heads(x @ lw["v_proj"].to(x.dtype).t(), self.num_kv_heads)
            return q, k, v

        self._warn_once(
            "qkv",
            "Attention Q/K/V projections not loaded; falling back to reshape-only "
            "(identity) attention. Outputs will be incorrect for real checkpoints.",
        )
        q = self._reshape_heads(x, self.num_heads)
        k = self._reshape_heads(x, self.num_kv_heads)
        v = self._reshape_heads(x, self.num_kv_heads)
        return q, k, v

    def _apply_out_proj(self, attn_output: torch.Tensor, layer_idx: int) -> torch.Tensor:
        w = self.layer_weights.get(layer_idx, {}).get("o_proj")
        if w is None:
            self._warn_once("o_proj", "Attention output projection missing; using identity fallback.")
            return attn_output
        return attn_output @ w.to(attn_output.dtype).t()

    def _for_attention(self, q, k, v):
        """Repeat KV heads to match Q head count for GQA/MQA."""
        if self.num_kv_heads == self.num_heads:
            return q, k, v
        reps = self.num_heads // self.num_kv_heads
        return q, k.repeat_interleave(reps, dim=1), v.repeat_interleave(reps, dim=1)

    def _reshape_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """[batch, seq, hidden] -> [batch, num_heads, seq, head_dim]."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape from [batch, seq_len, hidden_dim] to [batch, num_heads, seq_len, head_dim].
        """
        return self._reshape_heads(x, self.num_heads)

    def _reshape_from_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, hidden_dim].
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        # Transpose to [batch, seq_len, num_heads, head_dim]
        x = x.transpose(1, 2)
        # Reshape to [batch, seq_len, hidden_dim]
        return x.contiguous().view(batch_size, seq_len, self.hidden_dim)

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Manual attention computation (fallback if FlashAttention is not available).

        Args:
            q: Query [batch, num_heads, seq_len_q, head_dim]
            k: Key [batch, num_heads, seq_len_k, head_dim]
            v: Value [batch, num_heads, seq_len_v, head_dim]
            mask: Optional attention mask

        Returns:
            Attention output [batch, num_heads, seq_len_q, head_dim]
        """
        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores + mask

        # Apply causal mask for autoregressive generation
        seq_len_q = q.size(2)
        seq_len_k = k.size(2)
        if seq_len_q > 1:  # Only apply causal mask during prefill
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values: weights @ V
        attn_output = torch.matmul(attn_weights, v)

        return attn_output

    def _append_to_cache(
        self,
        kv_cache: Dict[str, torch.Tensor],
        token_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        DEPRECATED: This method is no longer used with pre-allocated buffers.

        The decode() method writes directly to the pre-allocated buffer,
        eliminating the need for this expensive copy operation.

        Kept for backward compatibility but should not be called.
        """
        raise DeprecationWarning(
            "_append_to_cache is deprecated. decode() uses pre-allocated buffers "
            "and writes in-place, eliminating this copy operation."
        )

    def clear_cache(self):
        """Clear KV cache to free memory (call between requests)."""
        # With caller-owned buffers we just reset our retained reference's
        # sequence length tracker; the buffer itself persists for reuse.
        if self._kv_cache_buffer is not None:
            self._kv_cache_buffer['seq_len'] = 0

    def get_cache_size_gb(self, seq_len: int) -> float:
        """
        Estimate KV cache memory usage in GB.

        Args:
            seq_len: Sequence length

        Returns:
            Memory usage in GB
        """
        # Each cache tensor: [batch=1, num_layers, seq_len, kv_heads, head_dim]
        # Two tensors (keys + values)
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        elements_per_cache = self.num_layers * seq_len * self.num_kv_heads * self.head_dim
        total_bytes = 2 * elements_per_cache * bytes_per_element  # keys + values
        return total_bytes / (1024 ** 3)
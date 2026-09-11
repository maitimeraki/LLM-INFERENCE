"""
Attention Engine for MoE Inference

Handles both prefill (full self-attention) and decode (incremental with KV cache) modes.
Optimized for low-latency token generation with efficient cache management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


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
        dtype: torch.dtype = torch.float16
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
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(head_dim)

        # Check if FlashAttention is available (torch >= 2.0)
        self.use_flash_attention = hasattr(F, 'scaled_dot_product_attention')

        # OPTIMIZATION: Pre-allocate KV cache buffers for max sequence length
        # This eliminates the need to recreate and copy tensors on every decode step
        # Memory cost: 2 * num_layers * max_seq_len * num_heads * head_dim * bytes_per_element
        # For 32 layers, 32 heads, 128 head_dim, 4096 tokens, fp16: ~2GB
        # This is a one-time allocation that persists across generation steps
        self._kv_cache_buffer = None
        self._cache_initialized = False

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Prefill phase: Process entire prompt with full self-attention.

        Args:
            input_ids: Input token embeddings [batch, seq_len, hidden_dim]
            attention_mask: Optional mask [batch, seq_len] (1 for valid, 0 for padding)

        Returns:
            hidden_states: Output embeddings [batch, seq_len, hidden_dim]
            kv_cache: Dictionary containing:
                - 'keys': Pre-allocated buffer [batch, num_layers, max_seq_len, num_heads, head_dim]
                - 'values': Pre-allocated buffer [batch, num_layers, max_seq_len, num_heads, head_dim]
                - 'seq_len': Current sequence length
        """
        batch_size, seq_len, hidden_dim = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Hidden dim {hidden_dim} does not match expected {self.hidden_dim}")

        # Move to device and cast to dtype
        input_ids = input_ids.to(self.device, dtype=self.dtype)

        # OPTIMIZATION: Initialize pre-allocated KV cache buffers (only once)
        if not self._cache_initialized or self._kv_cache_buffer is None:
            self._kv_cache_buffer = {
                'keys': torch.zeros(
                    batch_size, self.num_layers, self.max_seq_len, self.num_heads, self.head_dim,
                    device=self.device, dtype=self.dtype
                ),
                'values': torch.zeros(
                    batch_size, self.num_layers, self.max_seq_len, self.num_heads, self.head_dim,
                    device=self.device, dtype=self.dtype
                ),
                'seq_len': 0  # Track current position
            }
            self._cache_initialized = True

        # Reset buffer for new generation
        self._kv_cache_buffer['seq_len'] = 0

        # Reference to buffer (no copy)
        kv_cache = self._kv_cache_buffer

        # Process attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            # Convert to additive mask: 0 for valid positions, -inf for padding
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

        # Process through each layer
        hidden_states = input_ids

        for layer_idx in range(self.num_layers):
            # Compute Q, K, V for all tokens
            # In practice, these would come from layer-specific projection weights
            # For this engine, we assume input_ids are already post-embedding
            q = self._reshape_for_attention(hidden_states)  # [batch, num_heads, seq_len, head_dim]
            k = self._reshape_for_attention(hidden_states)  # [batch, num_heads, seq_len, head_dim]
            v = self._reshape_for_attention(hidden_states)  # [batch, num_heads, seq_len, head_dim]

            # Store K, V in cache for decode phase (OPTIMIZED: write directly to pre-allocated buffer)
            # Write to buffer slice [0:seq_len] - no allocation, just in-place write
            kv_cache['keys'][:, layer_idx, :seq_len] = k.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
            kv_cache['values'][:, layer_idx, :seq_len] = v.transpose(1, 2)

            # Compute attention
            if self.use_flash_attention:
                # Use PyTorch's optimized scaled_dot_product_attention
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=(attention_mask is None)  # Use causal mask if no explicit mask
                )
            else:
                # Manual attention computation
                attn_output = self._manual_attention(q, k, v, attention_mask)

            # Reshape back to [batch, seq_len, hidden_dim]
            hidden_states = self._reshape_from_attention(attn_output)

        # Update cache length tracker
        kv_cache['seq_len'] = seq_len

        return hidden_states, kv_cache

    def decode(
        self,
        token_id: torch.Tensor,
        kv_cache: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode phase: Process single token with cached KV from previous tokens.
        OPTIMIZED: Uses pre-allocated buffer, no tensor copying.

        Args:
            token_id: New token embedding [batch, 1, hidden_dim]
            kv_cache: Dictionary containing cached keys and values:
                - 'keys': Pre-allocated buffer [batch, num_layers, max_seq_len, num_heads, head_dim]
                - 'values': Pre-allocated buffer [batch, num_layers, max_seq_len, num_heads, head_dim]
                - 'seq_len': Current sequence length

        Returns:
            hidden_states: Output embedding for new token [batch, 1, hidden_dim]
            updated_kv_cache: Cache reference (same buffer, updated seq_len)
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

        # Process through each layer
        hidden_states = token_id

        for layer_idx in range(self.num_layers):
            # Compute Q, K, V for new token only
            q = self._reshape_for_attention(hidden_states)  # [batch, num_heads, 1, head_dim]
            k = self._reshape_for_attention(hidden_states)  # [batch, num_heads, 1, head_dim]
            v = self._reshape_for_attention(hidden_states)  # [batch, num_heads, 1, head_dim]

            # OPTIMIZATION: Write new K, V directly to buffer at current_pos (NO COPY!)
            # This is an in-place write to the pre-allocated buffer
            k_to_store = k.transpose(1, 2)  # [batch, 1, num_heads, head_dim]
            v_to_store = v.transpose(1, 2)

            kv_cache['keys'][:, layer_idx, current_pos:next_pos, :, :] = k_to_store.squeeze(1)
            kv_cache['values'][:, layer_idx, current_pos:next_pos, :, :] = v_to_store.squeeze(1)

            # OPTIMIZATION: Use slice view of cache (NO COPY!)
            # past_k/past_v are views into the buffer, not copies
            past_k = kv_cache['keys'][:, layer_idx, :next_pos, :, :]  # [batch, next_pos, num_heads, head_dim]
            past_v = kv_cache['values'][:, layer_idx, :next_pos, :, :]

            # Reshape for attention computation
            full_k = past_k.transpose(1, 2)  # [batch, num_heads, next_pos, head_dim]
            full_v = past_v.transpose(1, 2)

            # Compute attention (new token attends to all previous + itself)
            if self.use_flash_attention:
                attn_output = F.scaled_dot_product_attention(
                    q, full_k, full_v,
                    attn_mask=None,  # No mask needed, already causal by construction
                    dropout_p=0.0,
                    is_causal=False  # Already handled by cache structure
                )
            else:
                attn_output = self._manual_attention(q, full_k, full_v, mask=None)

            # Reshape back to [batch, 1, hidden_dim]
            hidden_states = self._reshape_from_attention(attn_output)

        # Update sequence length tracker (in-place update, no buffer recreation)
        kv_cache['seq_len'] = next_pos

        return hidden_states, kv_cache

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape from [batch, seq_len, hidden_dim] to [batch, num_heads, seq_len, head_dim].
        """
        batch_size, seq_len, _ = x.shape
        # Reshape to [batch, seq_len, num_heads, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose to [batch, num_heads, seq_len, head_dim]
        return x.transpose(1, 2)

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
        Manual attention computation (fallback if FlashAttention not available).

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

        The new decode() method writes directly to the pre-allocated buffer,
        eliminating the need for this expensive copy operation.

        Kept for backward compatibility but should not be called.
        """
        raise DeprecationWarning(
            "_append_to_cache is deprecated. The new decode() method uses "
            "pre-allocated buffers and writes in-place, eliminating this copy operation."
        )

    def clear_cache(self):
        """Clear KV cache to free memory (call between requests)."""
        # With pre-allocated buffers, we just reset the sequence length tracker
        # The buffer itself persists for reuse in the next generation
        if self._kv_cache_buffer is not None:
            self._kv_cache_buffer['seq_len'] = 0
            # Optional: zero out buffer to prevent any data leakage
            # self._kv_cache_buffer['keys'].zero_()
            # self._kv_cache_buffer['values'].zero_()

    def get_cache_size_gb(self, seq_len: int) -> float:
        """
        Estimate KV cache memory usage in GB.

        Args:
            seq_len: Sequence length

        Returns:
            Memory usage in GB
        """
        # Each cache tensor: [batch=1, num_layers, seq_len, num_heads, head_dim]
        # Two tensors (keys + values)
        bytes_per_element = 2 if self.dtype == torch.float16 else 4
        elements_per_cache = self.num_layers * seq_len * self.num_heads * self.head_dim
        total_bytes = 2 * elements_per_cache * bytes_per_element  # keys + values
        return total_bytes / (1024 ** 3)  # Convert to GB

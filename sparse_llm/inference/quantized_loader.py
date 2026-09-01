"""Load quantized experts and dequantize on-demand during inference.

Supports INT8 and FP8 quantization with synchronous dequantization.
Backward-compatible with non-quantized models.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from sparse_llm.models.quantization import QuantizationPolicy, extract_scale_names
from sparse_llm.cache import ExpertKey


class QuantizedExpertLoader:
    """Load and dequantize quantized experts on-demand.

    Handles:
    - Loading quantized weights from checkpoint
    - Extracting scale/zero-point tensors
    - Synchronous dequantization in INT8 or FP8 format
    - Backward compatibility with unquantized experts
    """

    def __init__(
        self,
        policy: QuantizationPolicy | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize quantized expert loader.

        Args:
            policy: QuantizationPolicy with format, group_size, scale_type, dequant_backend.
                   If None or format is None, loader operates in pass-through mode.
            device: Target device for dequantized tensors.
        """
        self.policy = policy
        self.device = torch.device(device) if isinstance(device, str) else device
        self._dequant_time_ms = 0.0

    def load_expert(
        self,
        key: ExpertKey,
        tensors: dict[str, torch.Tensor],
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Load expert tensors, dequantizing if policy specifies quantization.

        Args:
            key: (layer_index, expert_index) tuple for scale name extraction.
            tensors: Expert weight tensors from checkpoint (may be quantized).
            state_dict: Full model state dict for extracting scale/zero-point tensors.
                       Required if policy.is_quantized() is True.

        Returns:
            Dict of dequantized tensors, or original tensors if not quantized.
        """
        if not self.policy or not self.policy.is_quantized():
            # Pass-through: return original tensors unchanged
            return tensors

        if state_dict is None:
            raise ValueError(
                "state_dict required for quantized expert loading; "
                "scales/zero-points needed for dequantization"
            )

        layer_idx, expert_idx = key
        started = time.perf_counter()

        # Extract scale and zero-point tensor names for this expert
        scale_names = extract_scale_names(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            format=self.policy.format,
        )

        if not scale_names:
            # Format not recognized or not quantized; return original
            return tensors

        # Load scale and zero-point tensors from state_dict
        scale = state_dict.get(scale_names.get("scale"))
        zero_point = state_dict.get(scale_names.get("zero_point"))

        if scale is None:
            # Scale not found; assume unquantized
            return tensors

        # Dequantize each weight tensor
        dequantized = {}
        for name, tensor in tensors.items():
            if tensor.dtype in (torch.int8, torch.uint8):
                # INT8 dequantization: out = (q - zero) * scale
                dequantized[name] = self._dequantize_int8(
                    tensor, scale, zero_point
                )
            elif (
                hasattr(torch, "float8_e4m3fn") and tensor.dtype == torch.float8_e4m3fn
            ) or (
                hasattr(torch, "float8_e5m2") and tensor.dtype == torch.float8_e5m2
            ):
                # FP8 dequantization: convert to fp32 then scale
                dequantized[name] = self._dequantize_fp8(
                    tensor, scale
                )
            elif tensor.dtype in (torch.float16, torch.bfloat16):
                # Reduced-precision tensors: convert to fp32 then scale
                dequantized[name] = self._dequantize_reduced_precision(
                    tensor, scale
                )
            else:
                # Not quantized; pass through
                dequantized[name] = tensor

        elapsed = (time.perf_counter() - started) * 1000
        self._dequant_time_ms += elapsed
        return dequantized

    def _dequantize_int8(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dequantize INT8 quantized tensor.

        Formula: out = (q - zero_point) * scale
        """
        # Convert to float for dequantization
        q = quantized.to(torch.float32)

        if zero_point is not None:
            zp = zero_point.to(torch.float32)
            q = q - zp

        # Apply scale
        s = scale.to(torch.float32)
        dequantized = q * s

        # Move to target device
        return dequantized.to(self.device)

    def _dequantize_fp8(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize FP8 quantized tensor.

        FP8 is already in floating-point; just apply scale and move to device.
        """
        # Convert FP8 to float32 (automatic promotion)
        fp32 = quantized.to(torch.float32)

        # Apply scale
        s = scale.to(torch.float32)
        dequantized = fp32 * s

        # Move to target device
        return dequantized.to(self.device)

    def _dequantize_reduced_precision(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize reduced-precision (float16, bfloat16) tensor.

        Converts to float32, applies scale, and places on target device.
        """
        # Convert to float32
        fp32 = quantized.to(torch.float32)

        # Apply scale
        s = scale.to(torch.float32)
        dequantized = fp32 * s

        # Move to target device
        return dequantized.to(self.device)

    def dequantization_time_ms(self) -> float:
        """Return cumulative dequantization time in milliseconds."""
        return self._dequant_time_ms

    def reset_time(self) -> None:
        """Reset dequantization time counter."""
        self._dequant_time_ms = 0.0


__all__ = ["QuantizedExpertLoader"]

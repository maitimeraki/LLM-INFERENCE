import logging
from typing import Tuple
import torch
import numpy as np

logger = logging.getLogger(__name__)


class QuantizationManager:
    """Quantize and dequantize expert weights to int4."""

    def __init__(self, bits: int = 4):
        self.bits = bits
        if bits != 4:
            raise ValueError("Only 4-bit quantization supported")
        self.max_int = (1 << bits) - 1  # 15 for 4-bit

    def quantize(self, weights: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Quantize weights to int4.

        Returns:
            (quantized_weights: np.uint8, scales: torch.Tensor, zero_points: torch.Tensor)
        """
        weights_np = weights.cpu().detach().numpy()

        # Per-channel symmetric quantization
        channels = weights_np.shape[0] if len(weights_np.shape) > 1 else 1
        numel_per_channel = weights_np.size // channels

        scales = torch.zeros(channels, dtype=torch.float32)
        zero_points = torch.zeros(channels, dtype=torch.float32)
        quantized = np.zeros(weights_np.shape, dtype=np.uint8)

        for ch in range(channels):
            # Get channel weights
            if len(weights_np.shape) > 1:
                ch_weights = weights_np[ch, :].flatten()
            else:
                ch_weights = weights_np.flatten()

            # Compute scale and zero-point
            min_val = np.min(ch_weights)
            max_val = np.max(ch_weights)

            # Symmetric quantization: scale so [-max_abs, max_abs] -> [-8, 7]
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / (self.max_int // 2) if max_abs > 0 else 1.0
            zero_point = 0.0

            scales[ch] = scale
            zero_points[ch] = zero_point

            # Quantize: (x - zero) / scale
            quantized_ch = np.round((ch_weights - zero_point) / scale)
            quantized_ch = np.clip(quantized_ch, -self.max_int // 2, self.max_int // 2 - 1)

            # Store in uint8 (upper 4 bits for sign, lower 4 bits for magnitude)
            if len(weights_np.shape) > 1:
                quantized[ch, :] = quantized_ch.reshape(weights_np[ch, :].shape).astype(np.uint8)
            else:
                quantized = quantized_ch.astype(np.uint8)

        logger.debug(f"Quantized weights: shape={weights.shape}, scales={scales.mean():.6f}")
        return quantized, scales, zero_points

    def dequantize(self, quantized: np.ndarray,
                   scales: torch.Tensor,
                   zero_points: torch.Tensor,
                   target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Dequantize int4 weights to float.

        Args:
            quantized: quantized weights (np.uint8)
            scales: scale factors per channel
            zero_points: zero points per channel
            target_dtype: output dtype

        Returns:
            dequantized weights (torch.Tensor)
        """
        quantized_float = torch.from_numpy(quantized.astype(np.float32))

        # Reshape for broadcasting
        if len(quantized_float.shape) > 1:
            scales = scales.view(-1, 1)
            zero_points = zero_points.view(-1, 1)

        # Dequantize: x_original = (x_int - zero) * scale
        dequantized = (quantized_float - zero_points) * scales

        return dequantized.to(target_dtype)

    def quantization_ratio(self, original_bytes: int) -> float:
        """Compute compression ratio."""
        # Original: float32 = 4 bytes per value
        # Quantized: 4 bits per value = 0.5 bytes per value (2 values per byte)
        # Scales/zero_points: float32 per channel (~128 channels for typical weights)
        quantized_bytes = (original_bytes / 4) * 0.5 + (128 * 4)  # scales + zero_points
        return original_bytes / quantized_bytes

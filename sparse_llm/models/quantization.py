"""Quantization policy and format detection for model loading and inference.

Config-based detection of quantization formats from checkpoint metadata.
No actual model loading — policy inference and scale/zero-point extraction only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuantizationPolicy:
    """Quantization configuration for a model or layer.

    Attributes:
        format: Quantization format (bfloat16, fp32, int8, fp8, or None for unquantized).
        group_size: Grouping size for quantization scales (-1 for per-channel, per-token).
        scale_type: Type of scale parameter (absmax, minmax, or None).
        dequant_backend: Backend for dequantization (aten, triton, or None for native).
    """

    format: str | None = None
    group_size: int = -1
    scale_type: str | None = None
    dequant_backend: str | None = None

    def __post_init__(self) -> None:
        """Validate quantization policy fields."""
        if self.format is not None:
            valid_formats = {"bfloat16", "fp32", "int8", "fp8"}
            if self.format not in valid_formats:
                raise ValueError(
                    f"unsupported quantization format {self.format!r}, "
                    f"must be one of {valid_formats} or None"
                )

        if not isinstance(self.group_size, int):
            raise TypeError(f"group_size must be int, got {type(self.group_size).__name__}")

        if self.scale_type is not None:
            valid_scale_types = {"absmax", "minmax"}
            if self.scale_type not in valid_scale_types:
                raise ValueError(
                    f"unsupported scale_type {self.scale_type!r}, "
                    f"must be one of {valid_scale_types} or None"
                )

        if self.dequant_backend is not None:
            valid_backends = {"aten", "triton"}
            if self.dequant_backend not in valid_backends:
                raise ValueError(
                    f"unsupported dequant_backend {self.dequant_backend!r}, "
                    f"must be one of {valid_backends} or None"
                )

    def is_quantized(self) -> bool:
        """Return True if a quantization format is active."""
        return self.format is not None

    def is_int8(self) -> bool:
        """Return True if using INT8 quantization."""
        return self.format == "int8"

    def is_fp8(self) -> bool:
        """Return True if using FP8 quantization."""
        return self.format == "fp8"


def detect_quantization(config: Any | None) -> QuantizationPolicy | None:
    """Detect quantization policy from checkpoint configuration.

    Scans configuration for quantization metadata and returns a policy
    if quantization is detected, otherwise returns None.

    Supports detection of:
    - INT8 quantization (quantization_config.quant_method == 'bquan' or similar)
    - FP8 quantization (quantization_config.quant_method == 'fp8' or similar)
    - Format inference from config keys and values

    Args:
        config: Model configuration object (dict-like or namespace).
               If None, returns None (unquantized).

    Returns:
        QuantizationPolicy if quantization is detected, else None.
    """
    if config is None:
        return None

    # Check for quantization_config attribute (transformers convention)
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        return None

    # Extract quantization method and parameters
    quant_method = getattr(quant_config, "quant_method", None)
    if quant_method is None:
        # Fallback: check for common quantization config keys
        quant_method = getattr(quant_config, "quantization_method", None)

    if quant_method is None:
        return None

    # Normalize method name
    method_lower = quant_method.lower() if isinstance(quant_method, str) else ""

    # Detect INT8
    if method_lower in {"int8", "bquan", "bquant"}:
        group_size = getattr(quant_config, "group_size", -1)
        scale_type = getattr(quant_config, "scale_type", "absmax")
        dequant_backend = getattr(quant_config, "dequant_backend", None)
        return QuantizationPolicy(
            format="int8",
            group_size=group_size,
            scale_type=scale_type,
            dequant_backend=dequant_backend,
        )

    # Detect FP8
    if method_lower in {"fp8", "float8"}:
        group_size = getattr(quant_config, "group_size", -1)
        scale_type = getattr(quant_config, "scale_type", "absmax")
        dequant_backend = getattr(quant_config, "dequant_backend", None)
        return QuantizationPolicy(
            format="fp8",
            group_size=group_size,
            scale_type=scale_type,
            dequant_backend=dequant_backend,
        )

    # Detect other formats (bfloat16, fp32) if explicitly marked
    if method_lower in {"bfloat16", "bf16"}:
        return QuantizationPolicy(format="bfloat16")

    if method_lower in {"fp32", "float32"}:
        return QuantizationPolicy(format="fp32")

    # Unknown quantization method: return None (treat as unquantized)
    return None


def extract_scale_names(
    layer_idx: int,
    expert_idx: int | None = None,
    format: str | None = None,
) -> dict[str, str]:
    """Extract expected scale and zero-point tensor names for a layer or expert.

    Returns a mapping of semantic names to tensor names based on quantization
    format and layer/expert indices. Used for locating scale tensors in
    checkpoints and state dicts.

    Args:
        layer_idx: Layer index in the model.
        expert_idx: Expert index within the layer (if layer is MoE).
                   If None, assumes dense layer.
        format: Quantization format (int8, fp8, etc).
               If None or unquantized format, returns empty dict.

    Returns:
        Dict mapping semantic names (scale, zero_point) to checkpoint tensor names.
        Empty dict if format is None or unsupported.

    Example:
        >>> extract_scale_names(5, expert_idx=2, format="int8")
        {
            'scale': 'model.layers.5.mlp.experts.2.w_q_scale',
            'zero_point': 'model.layers.5.mlp.experts.2.w_q_zero'
        }
    """
    if format is None or format not in {"int8", "fp8"}:
        return {}

    scale_names = {}

    if expert_idx is not None:
        # MoE layer: format is model.layers.{layer}.experts.{expert}.{param}_{format}_scale
        scale_names["scale"] = f"model.layers.{layer_idx}.experts.{expert_idx}.w_q_scale"
        scale_names["zero_point"] = (
            f"model.layers.{layer_idx}.experts.{expert_idx}.w_q_zero"
        )
    else:
        # Dense layer: format is model.layers.{layer}.{param}_{format}_scale
        scale_names["scale"] = f"model.layers.{layer_idx}.w_q_scale"
        scale_names["zero_point"] = f"model.layers.{layer_idx}.w_q_zero"

    return scale_names


__all__ = [
    "QuantizationPolicy",
    "detect_quantization",
    "extract_scale_names",
]

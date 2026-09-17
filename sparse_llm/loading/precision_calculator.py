"""Precision-aware byte calculations for model weights.

Supports FP32, FP16, INT8, and INT4 precision formats with accurate
bytes-per-parameter calculations.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass


class Precision(str, Enum):
    """Supported precision formats for model weights."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def bytes_per_param(self) -> float:
        """Get bytes per parameter for this precision format.

        Returns:
            Bytes per parameter (FP32=4, FP16=2, INT8=1, INT4=0.5)
        """
        return {
            Precision.FP32: 4.0,
            Precision.FP16: 2.0,
            Precision.INT8: 1.0,
            Precision.INT4: 0.5,
        }[self]

    @classmethod
    def from_string(cls, value: str) -> Precision:
        """Parse precision from string (case-insensitive).

        Args:
            value: Precision string (e.g., "fp16", "FP16", "float16")

        Returns:
            Precision enum value

        Raises:
            ValueError: If precision string is not recognized
        """
        value_lower = value.lower()

        # Handle common aliases
        aliases = {
            "fp32": cls.FP32,
            "float32": cls.FP32,
            "f32": cls.FP32,
            "fp16": cls.FP16,
            "float16": cls.FP16,
            "f16": cls.FP16,
            "half": cls.FP16,
            "int8": cls.INT8,
            "i8": cls.INT8,
            "int4": cls.INT4,
            "i4": cls.INT4,
        }

        if value_lower in aliases:
            return aliases[value_lower]

        raise ValueError(
            f"Unsupported precision: {value}. "
            f"Supported: fp32, fp16, int8, int4"
        )


@dataclass(frozen=True)
class PrecisionMemoryProfile:
    """Memory profile for a specific precision format.

    Attributes:
        precision: The precision format
        bytes_per_param: Bytes per parameter for this precision
        total_params: Total number of parameters
        total_bytes: Total memory required in bytes
    """
    precision: Precision
    bytes_per_param: float
    total_params: int
    total_bytes: int

    @property
    def total_gb(self) -> float:
        """Total memory in gigabytes."""
        return self.total_bytes / (1024 ** 3)

    @property
    def total_mb(self) -> float:
        """Total memory in megabytes."""
        return self.total_bytes / (1024 ** 2)


class PrecisionCalculator:
    """Calculate memory requirements based on precision format."""

    def calculate_bytes(self, num_params: int, precision: Precision) -> int:
        """Calculate total bytes for given parameters and precision.

        Args:
            num_params: Number of parameters
            precision: Target precision format

        Returns:
            Total bytes required
        """
        bytes_per_param = precision.bytes_per_param
        total_bytes = int(num_params * bytes_per_param)
        return total_bytes

    def create_memory_profile(
        self,
        num_params: int,
        precision: Precision
    ) -> PrecisionMemoryProfile:
        """Create a complete memory profile for given parameters.

        Args:
            num_params: Number of parameters
            precision: Target precision format

        Returns:
            PrecisionMemoryProfile with detailed memory breakdown
        """
        bytes_per_param = precision.bytes_per_param
        total_bytes = self.calculate_bytes(num_params, precision)

        return PrecisionMemoryProfile(
            precision=precision,
            bytes_per_param=bytes_per_param,
            total_params=num_params,
            total_bytes=total_bytes
        )

    def compare_precisions(
        self,
        num_params: int
    ) -> dict[Precision, PrecisionMemoryProfile]:
        """Compare memory requirements across all precision formats.

        Args:
            num_params: Number of parameters

        Returns:
            Dictionary mapping precision to memory profile
        """
        profiles = {}
        for precision in Precision:
            profiles[precision] = self.create_memory_profile(num_params, precision)
        return profiles

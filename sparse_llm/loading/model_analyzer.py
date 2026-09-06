"""Precision-aware model analysis combining architecture introspection with memory calculations.

This module extends ModelIntrospector to support user-configurable precision formats,
enabling accurate memory planning for FP32, FP16, INT8, and INT4 inference.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from sparse_llm.loading.model_introspector import ModelIntrospector, ModelInfo
from sparse_llm.loading.precision_calculator import Precision, PrecisionCalculator


@dataclass(frozen=True)
class ModelAnalysis:
    """Complete model analysis with precision-aware memory calculations.

    Attributes:
        model_info: Basic model architecture information
        precision: Target precision format
        shared_weight_bytes: Bytes for shared/per-token weights at target precision
        expert_weight_bytes: Bytes per expert at target precision
        total_bytes: Total model size at target precision
        per_token_active_bytes: Bytes active per forward pass (shared + active experts)
    """
    model_info: ModelInfo
    precision: Precision
    shared_weight_bytes: int
    expert_weight_bytes: int  # Per expert
    total_bytes: int
    per_token_active_bytes: int  # Shared + (num_experts_per_tok * expert_weight_bytes * num_layers)

    @property
    def total_gb(self) -> float:
        """Total model size in gigabytes."""
        return self.total_bytes / (1024 ** 3)

    @property
    def shared_weight_gb(self) -> float:
        """Shared weights size in gigabytes."""
        return self.shared_weight_bytes / (1024 ** 3)

    @property
    def expert_weight_mb(self) -> float:
        """Per-expert weight size in megabytes."""
        return self.expert_weight_bytes / (1024 ** 2)

    @property
    def per_token_active_gb(self) -> float:
        """Active memory per token in gigabytes."""
        return self.per_token_active_bytes / (1024 ** 3)


class ModelAnalyzer:
    """Analyze models with precision-aware memory calculations.

    Combines ModelIntrospector (architecture detection) with PrecisionCalculator
    (memory calculations) to provide complete model analysis at any precision.
    """

    def __init__(self):
        """Initialize analyzer with introspector and calculator."""
        self.introspector = ModelIntrospector()
        self.calculator = PrecisionCalculator()

    def analyze(
        self,
        model_id: str,
        precision: Precision | str = Precision.FP16
    ) -> ModelAnalysis:
        """Analyze model at specified precision.

        Args:
            model_id: Hugging Face model ID or local path
            precision: Target precision format (FP32/FP16/INT8/INT4)

        Returns:
            ModelAnalysis with precision-aware memory calculations
        """
        # Parse precision if string
        if isinstance(precision, str):
            precision = Precision.from_string(precision)

        # Get base model info (uses FP16 hardcoded calculations)
        model_info = self.introspector.introspect(model_id)

        # Recalculate with target precision
        return self.analyze_from_model_info(model_info, precision)

    def analyze_from_config(
        self,
        config: Any,
        model_id: str,
        precision: Precision | str = Precision.FP16
    ) -> ModelAnalysis:
        """Analyze from loaded config at specified precision.

        Args:
            config: Transformers config object
            model_id: Model identifier for logging
            precision: Target precision format

        Returns:
            ModelAnalysis with precision-aware memory calculations
        """
        # Parse precision if string
        if isinstance(precision, str):
            precision = Precision.from_string(precision)

        # Get base model info
        model_info = self.introspector.introspect_from_config(config, model_id)

        # Recalculate with target precision
        return self.analyze_from_model_info(model_info, precision)

    def analyze_from_model_info(
        self,
        model_info: ModelInfo,
        precision: Precision
    ) -> ModelAnalysis:
        """Convert ModelInfo to precision-aware ModelAnalysis.

        Args:
            model_info: Base model information (calculated at FP16)
            precision: Target precision format

        Returns:
            ModelAnalysis recalculated at target precision
        """
        # ModelInfo uses FP16 (2 bytes per param), so we need to scale
        fp16_bytes_per_param = 2.0
        target_bytes_per_param = precision.bytes_per_param
        scale_factor = target_bytes_per_param / fp16_bytes_per_param

        # Scale all byte measurements
        shared_weight_bytes = int(model_info.shared_weight_bytes * scale_factor)
        expert_weight_bytes = int(model_info.expert_weight_bytes * scale_factor)
        total_bytes = int(model_info.total_bytes * scale_factor)

        # Calculate per-token active bytes (shared + active experts per layer)
        if model_info.is_moe:
            per_token_active_bytes = shared_weight_bytes + (
                model_info.num_experts_per_tok * expert_weight_bytes * model_info.num_layers
            )
        else:
            # Dense model: all weights are always active
            per_token_active_bytes = total_bytes

        return ModelAnalysis(
            model_info=model_info,
            precision=precision,
            shared_weight_bytes=shared_weight_bytes,
            expert_weight_bytes=expert_weight_bytes,
            total_bytes=total_bytes,
            per_token_active_bytes=per_token_active_bytes
        )

    def compare_precisions(
        self,
        model_id: str
    ) -> dict[Precision, ModelAnalysis]:
        """Compare memory requirements across all precision formats.

        Args:
            model_id: Hugging Face model ID or local path

        Returns:
            Dictionary mapping precision to model analysis
        """
        # Get base model info once
        model_info = self.introspector.introspect(model_id)

        # Generate analysis for each precision
        analyses = {}
        for precision in Precision:
            analyses[precision] = self.analyze_from_model_info(model_info, precision)

        return analyses

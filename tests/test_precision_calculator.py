"""Tests for precision-aware memory calculations."""

import pytest
from sparse_llm.loading.precision_calculator import (
    Precision,
    PrecisionCalculator,
    PrecisionMemoryProfile,
)


class TestPrecision:
    """Test Precision enum and its methods."""

    def test_bytes_per_param_fp32(self):
        """FP32 should be 4 bytes per parameter."""
        assert Precision.FP32.bytes_per_param == 4.0

    def test_bytes_per_param_fp16(self):
        """FP16 should be 2 bytes per parameter."""
        assert Precision.FP16.bytes_per_param == 2.0

    def test_bytes_per_param_int8(self):
        """INT8 should be 1 byte per parameter."""
        assert Precision.INT8.bytes_per_param == 1.0

    def test_bytes_per_param_int4(self):
        """INT4 should be 0.5 bytes per parameter."""
        assert Precision.INT4.bytes_per_param == 0.5

    def test_from_string_lowercase(self):
        """Should parse lowercase precision strings."""
        assert Precision.from_string("fp32") == Precision.FP32
        assert Precision.from_string("fp16") == Precision.FP16
        assert Precision.from_string("int8") == Precision.INT8
        assert Precision.from_string("int4") == Precision.INT4

    def test_from_string_uppercase(self):
        """Should parse uppercase precision strings."""
        assert Precision.from_string("FP32") == Precision.FP32
        assert Precision.from_string("FP16") == Precision.FP16
        assert Precision.from_string("INT8") == Precision.INT8
        assert Precision.from_string("INT4") == Precision.INT4

    def test_from_string_aliases(self):
        """Should handle common aliases."""
        # FP32 aliases
        assert Precision.from_string("float32") == Precision.FP32
        assert Precision.from_string("f32") == Precision.FP32

        # FP16 aliases
        assert Precision.from_string("float16") == Precision.FP16
        assert Precision.from_string("f16") == Precision.FP16
        assert Precision.from_string("half") == Precision.FP16

        # INT8 aliases
        assert Precision.from_string("i8") == Precision.INT8

        # INT4 aliases
        assert Precision.from_string("i4") == Precision.INT4

    def test_from_string_invalid(self):
        """Should raise ValueError for invalid precision."""
        with pytest.raises(ValueError, match="Unsupported precision"):
            Precision.from_string("fp8")

        with pytest.raises(ValueError, match="Unsupported precision"):
            Precision.from_string("invalid")


class TestPrecisionMemoryProfile:
    """Test PrecisionMemoryProfile dataclass."""

    def test_total_gb(self):
        """Should correctly convert bytes to gigabytes."""
        profile = PrecisionMemoryProfile(
            precision=Precision.FP16,
            bytes_per_param=2.0,
            total_params=1_000_000_000,  # 1B params
            total_bytes=2_000_000_000,  # 2GB
        )
        assert abs(profile.total_gb - 1.862645) < 0.0001  # 2B / (1024^3)

    def test_total_mb(self):
        """Should correctly convert bytes to megabytes."""
        profile = PrecisionMemoryProfile(
            precision=Precision.FP16,
            bytes_per_param=2.0,
            total_params=1_000_000,  # 1M params
            total_bytes=2_000_000,  # 2MB
        )
        assert abs(profile.total_mb - 1.907348) < 0.0001  # 2M / (1024^2)


class TestPrecisionCalculator:
    """Test PrecisionCalculator calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance for tests."""
        return PrecisionCalculator()

    def test_calculate_bytes_fp32(self, calculator):
        """Should calculate correct bytes for FP32."""
        # 1B parameters at FP32 (4 bytes each) = 4GB
        num_params = 1_000_000_000
        result = calculator.calculate_bytes(num_params, Precision.FP32)
        assert result == 4_000_000_000

    def test_calculate_bytes_fp16(self, calculator):
        """Should calculate correct bytes for FP16."""
        # 1B parameters at FP16 (2 bytes each) = 2GB
        num_params = 1_000_000_000
        result = calculator.calculate_bytes(num_params, Precision.FP16)
        assert result == 2_000_000_000

    def test_calculate_bytes_int8(self, calculator):
        """Should calculate correct bytes for INT8."""
        # 1B parameters at INT8 (1 byte each) = 1GB
        num_params = 1_000_000_000
        result = calculator.calculate_bytes(num_params, Precision.INT8)
        assert result == 1_000_000_000

    def test_calculate_bytes_int4(self, calculator):
        """Should calculate correct bytes for INT4."""
        # 1B parameters at INT4 (0.5 bytes each) = 0.5GB
        num_params = 1_000_000_000
        result = calculator.calculate_bytes(num_params, Precision.INT4)
        assert result == 500_000_000

    def test_calculate_bytes_small_model(self, calculator):
        """Should handle small models correctly."""
        # 100M parameters at FP16 = 200MB
        num_params = 100_000_000
        result = calculator.calculate_bytes(num_params, Precision.FP16)
        assert result == 200_000_000

    def test_create_memory_profile(self, calculator):
        """Should create complete memory profile."""
        num_params = 1_000_000_000  # 1B params
        profile = calculator.create_memory_profile(num_params, Precision.FP16)

        assert profile.precision == Precision.FP16
        assert profile.bytes_per_param == 2.0
        assert profile.total_params == 1_000_000_000
        assert profile.total_bytes == 2_000_000_000
        assert abs(profile.total_gb - 1.862645) < 0.0001

    def test_compare_precisions(self, calculator):
        """Should compare all precision formats."""
        num_params = 1_000_000_000  # 1B params
        profiles = calculator.compare_precisions(num_params)

        # Check all precisions are present
        assert Precision.FP32 in profiles
        assert Precision.FP16 in profiles
        assert Precision.INT8 in profiles
        assert Precision.INT4 in profiles

        # Check bytes are correct
        assert profiles[Precision.FP32].total_bytes == 4_000_000_000
        assert profiles[Precision.FP16].total_bytes == 2_000_000_000
        assert profiles[Precision.INT8].total_bytes == 1_000_000_000
        assert profiles[Precision.INT4].total_bytes == 500_000_000

    def test_compare_precisions_ratios(self, calculator):
        """Should maintain correct ratios between precisions."""
        num_params = 1_000_000_000  # 1B params
        profiles = calculator.compare_precisions(num_params)

        fp32_bytes = profiles[Precision.FP32].total_bytes
        fp16_bytes = profiles[Precision.FP16].total_bytes
        int8_bytes = profiles[Precision.INT8].total_bytes
        int4_bytes = profiles[Precision.INT4].total_bytes

        # FP32 should be 2x FP16
        assert fp32_bytes == 2 * fp16_bytes

        # FP16 should be 2x INT8
        assert fp16_bytes == 2 * int8_bytes

        # INT8 should be 2x INT4
        assert int8_bytes == 2 * int4_bytes

        # FP32 should be 8x INT4
        assert fp32_bytes == 8 * int4_bytes


class TestRealWorldScenarios:
    """Test real-world model size calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance for tests."""
        return PrecisionCalculator()

    def test_mixtral_8x7b_fp16(self, calculator):
        """Test Mixtral 8x7B at FP16 (approximate)."""
        # Mixtral has ~46.7B total parameters
        num_params = 46_700_000_000
        result = calculator.calculate_bytes(num_params, Precision.FP16)

        # Should be approximately 93.4GB
        expected = 93_400_000_000
        assert abs(result - expected) < 1_000_000  # Within 1MB tolerance

    def test_mixtral_8x7b_int8(self, calculator):
        """Test Mixtral 8x7B at INT8 (approximate)."""
        num_params = 46_700_000_000
        result = calculator.calculate_bytes(num_params, Precision.INT8)

        # Should be approximately 46.7GB
        expected = 46_700_000_000
        assert abs(result - expected) < 1_000_000

    def test_llama2_7b_fp16(self, calculator):
        """Test LLaMA 2 7B at FP16."""
        num_params = 7_000_000_000
        result = calculator.calculate_bytes(num_params, Precision.FP16)

        # Should be approximately 14GB
        expected = 14_000_000_000
        assert abs(result - expected) < 1_000_000

    def test_llama2_7b_int4(self, calculator):
        """Test LLaMA 2 7B at INT4."""
        num_params = 7_000_000_000
        result = calculator.calculate_bytes(num_params, Precision.INT4)

        # Should be approximately 3.5GB
        expected = 3_500_000_000
        assert abs(result - expected) < 1_000_000

    def test_expert_weight_calculation(self, calculator):
        """Test expert weight calculation for MoE models."""
        # Mixtral expert: approximately 437M parameters each
        expert_params = 437_000_000

        # At FP16: ~874MB per expert
        fp16_bytes = calculator.calculate_bytes(expert_params, Precision.FP16)
        assert abs(fp16_bytes - 874_000_000) < 1_000_000

        # At INT8: ~437MB per expert
        int8_bytes = calculator.calculate_bytes(expert_params, Precision.INT8)
        assert abs(int8_bytes - 437_000_000) < 1_000_000

        # At INT4: ~218.5MB per expert
        int4_bytes = calculator.calculate_bytes(expert_params, Precision.INT4)
        assert abs(int4_bytes - 218_500_000) < 1_000_000

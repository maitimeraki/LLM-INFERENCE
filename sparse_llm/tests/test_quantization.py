"""Tests for quantization policy and format detection."""

from types import SimpleNamespace

import pytest

from sparse_llm.models.quantization import (
    QuantizationPolicy,
    detect_quantization,
    extract_scale_names,
)


class TestQuantizationPolicy:
    """QuantizationPolicy dataclass tests."""

    def test_default_policy_is_unquantized(self):
        """Default policy with no format specified is unquantized."""
        policy = QuantizationPolicy()
        assert not policy.is_quantized()
        assert policy.format is None
        assert policy.group_size == -1
        assert policy.scale_type is None
        assert policy.dequant_backend is None

    def test_policy_is_frozen(self):
        """QuantizationPolicy instances are immutable."""
        policy = QuantizationPolicy(format="int8")
        with pytest.raises(AttributeError):
            policy.format = "fp8"

    def test_int8_policy_creation(self):
        """Create INT8 quantization policy."""
        policy = QuantizationPolicy(
            format="int8",
            group_size=32,
            scale_type="absmax",
            dequant_backend="triton",
        )
        assert policy.is_quantized()
        assert policy.is_int8()
        assert not policy.is_fp8()
        assert policy.format == "int8"
        assert policy.group_size == 32
        assert policy.scale_type == "absmax"
        assert policy.dequant_backend == "triton"

    def test_fp8_policy_creation(self):
        """Create FP8 quantization policy."""
        policy = QuantizationPolicy(format="fp8", group_size=-1, scale_type="minmax")
        assert policy.is_quantized()
        assert policy.is_fp8()
        assert not policy.is_int8()
        assert policy.format == "fp8"

    def test_bfloat16_policy_creation(self):
        """Create bfloat16 policy."""
        policy = QuantizationPolicy(format="bfloat16")
        assert policy.is_quantized()
        assert policy.format == "bfloat16"

    def test_fp32_policy_creation(self):
        """Create fp32 policy."""
        policy = QuantizationPolicy(format="fp32")
        assert policy.is_quantized()
        assert policy.format == "fp32"

    def test_invalid_format_raises_error(self):
        """Invalid quantization format raises ValueError."""
        with pytest.raises(ValueError, match="unsupported quantization format"):
            QuantizationPolicy(format="invalid")

    def test_non_int_group_size_raises_error(self):
        """Non-integer group_size raises TypeError."""
        with pytest.raises(TypeError, match="group_size must be int"):
            QuantizationPolicy(group_size=32.5)

    def test_invalid_scale_type_raises_error(self):
        """Invalid scale_type raises ValueError."""
        with pytest.raises(ValueError, match="unsupported scale_type"):
            QuantizationPolicy(format="int8", scale_type="invalid")

    def test_invalid_dequant_backend_raises_error(self):
        """Invalid dequant_backend raises ValueError."""
        with pytest.raises(ValueError, match="unsupported dequant_backend"):
            QuantizationPolicy(format="int8", dequant_backend="invalid")

    def test_valid_scale_types(self):
        """Valid scale types are accepted."""
        for scale_type in ["absmax", "minmax"]:
            policy = QuantizationPolicy(format="int8", scale_type=scale_type)
            assert policy.scale_type == scale_type

    def test_valid_dequant_backends(self):
        """Valid dequant backends are accepted."""
        for backend in ["aten", "triton"]:
            policy = QuantizationPolicy(format="int8", dequant_backend=backend)
            assert policy.dequant_backend == backend


class TestDetectQuantization:
    """detect_quantization function tests."""

    def test_none_config_returns_none(self):
        """None config returns None."""
        result = detect_quantization(None)
        assert result is None

    def test_config_without_quantization_config_returns_none(self):
        """Config without quantization_config returns None."""
        config = SimpleNamespace(model_type="fake", num_hidden_layers=12)
        result = detect_quantization(config)
        assert result is None

    def test_quantization_config_without_method_returns_none(self):
        """quantization_config without method returns None."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                group_size=32,
                scale_type="absmax",
            )
        )
        result = detect_quantization(config)
        assert result is None

    def test_detect_int8_quantization(self):
        """Detect INT8 quantization from config."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="int8",
                group_size=32,
                scale_type="absmax",
                dequant_backend="triton",
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "int8"
        assert result.group_size == 32
        assert result.scale_type == "absmax"
        assert result.dequant_backend == "triton"

    def test_detect_int8_with_bquan_method(self):
        """Detect INT8 with 'bquan' method variant."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="bquan",
                group_size=64,
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "int8"
        assert result.group_size == 64

    def test_detect_int8_with_bquant_method(self):
        """Detect INT8 with 'bquant' method variant."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="bquant",
                group_size=128,
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "int8"
        assert result.group_size == 128

    def test_detect_fp8_quantization(self):
        """Detect FP8 quantization from config."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="fp8",
                group_size=-1,
                scale_type="minmax",
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "fp8"
        assert result.group_size == -1
        assert result.scale_type == "minmax"

    def test_detect_fp8_with_float8_method(self):
        """Detect FP8 with 'float8' method variant."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="float8",
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "fp8"

    def test_detect_bfloat16_quantization(self):
        """Detect bfloat16 format."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="bfloat16",
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "bfloat16"

    def test_detect_bf16_quantization(self):
        """Detect bfloat16 with 'bf16' variant."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="bf16",
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "bfloat16"

    def test_detect_fp32_quantization(self):
        """Detect fp32 format."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="fp32",
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "fp32"

    def test_detect_float32_quantization(self):
        """Detect fp32 with 'float32' variant."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="float32",
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "fp32"

    def test_unknown_method_returns_none(self):
        """Unknown quantization method returns None."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quant_method="unknown_format",
            )
        )
        result = detect_quantization(config)
        assert result is None

    def test_case_insensitive_detection(self):
        """Quantization method detection is case-insensitive."""
        for method in ["INT8", "Int8", "iNt8"]:
            config = SimpleNamespace(
                quantization_config=SimpleNamespace(quant_method=method)
            )
            result = detect_quantization(config)
            assert result is not None
            assert result.format == "int8"

    def test_detect_with_missing_optional_fields(self):
        """Detection works with missing optional fields (uses defaults)."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(quant_method="int8")
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "int8"
        assert result.group_size == -1
        assert result.scale_type == "absmax"
        assert result.dequant_backend is None

    def test_detect_with_fallback_quantization_method(self):
        """Detection works with 'quantization_method' fallback attribute."""
        config = SimpleNamespace(
            quantization_config=SimpleNamespace(
                quantization_method="int8",
                group_size=32,
            )
        )
        result = detect_quantization(config)
        assert result is not None
        assert result.format == "int8"
        assert result.group_size == 32


class TestExtractScaleNames:
    """extract_scale_names function tests."""

    def test_unquantized_format_returns_empty(self):
        """Unquantized format returns empty dict."""
        result = extract_scale_names(layer_idx=0, format=None)
        assert result == {}

    def test_unsupported_format_returns_empty(self):
        """Unsupported format returns empty dict."""
        result = extract_scale_names(layer_idx=0, format="bfloat16")
        assert result == {}

    def test_dense_int8_scale_names(self):
        """Extract scale names for dense INT8 layer."""
        result = extract_scale_names(layer_idx=5, format="int8")
        assert result == {
            "scale": "model.layers.5.w_q_scale",
            "zero_point": "model.layers.5.w_q_zero",
        }

    def test_expert_int8_scale_names(self):
        """Extract scale names for MoE expert INT8 layer."""
        result = extract_scale_names(layer_idx=5, expert_idx=2, format="int8")
        assert result == {
            "scale": "model.layers.5.experts.2.w_q_scale",
            "zero_point": "model.layers.5.experts.2.w_q_zero",
        }

    def test_dense_fp8_scale_names(self):
        """Extract scale names for dense FP8 layer."""
        result = extract_scale_names(layer_idx=10, format="fp8")
        assert result == {
            "scale": "model.layers.10.w_q_scale",
            "zero_point": "model.layers.10.w_q_zero",
        }

    def test_expert_fp8_scale_names(self):
        """Extract scale names for MoE expert FP8 layer."""
        result = extract_scale_names(layer_idx=10, expert_idx=7, format="fp8")
        assert result == {
            "scale": "model.layers.10.experts.7.w_q_scale",
            "zero_point": "model.layers.10.experts.7.w_q_zero",
        }

    def test_layer_indices_reflected_in_names(self):
        """Scale names correctly reflect layer indices."""
        for layer_idx in [0, 5, 31]:
            result = extract_scale_names(layer_idx=layer_idx, format="int8")
            assert f"model.layers.{layer_idx}" in result["scale"]
            assert f"model.layers.{layer_idx}" in result["zero_point"]

    def test_expert_indices_reflected_in_names(self):
        """Scale names correctly reflect expert indices."""
        for expert_idx in [0, 2, 7]:
            result = extract_scale_names(
                layer_idx=5, expert_idx=expert_idx, format="int8"
            )
            assert f"experts.{expert_idx}" in result["scale"]
            assert f"experts.{expert_idx}" in result["zero_point"]

    def test_zero_indices_handled_correctly(self):
        """Zero indices (layer=0, expert=0) are handled correctly."""
        result = extract_scale_names(layer_idx=0, expert_idx=0, format="int8")
        assert result == {
            "scale": "model.layers.0.experts.0.w_q_scale",
            "zero_point": "model.layers.0.experts.0.w_q_zero",
        }


def test_backward_compatibility_unquantized_models():
    """Unquantized models (no quantization_config) remain unaffected."""
    # Model without quantization config
    config = SimpleNamespace(
        model_type="llama",
        num_hidden_layers=32,
        num_attention_heads=32,
    )
    result = detect_quantization(config)
    assert result is None


def test_quantization_policy_is_hashable():
    """QuantizationPolicy instances are hashable (frozen dataclass)."""
    policy1 = QuantizationPolicy(format="int8", group_size=32)
    policy2 = QuantizationPolicy(format="int8", group_size=32)
    policy3 = QuantizationPolicy(format="fp8")

    # Should be able to use in sets and as dict keys
    policy_set = {policy1, policy2, policy3}
    assert len(policy_set) == 2  # policy1 and policy2 are equal

    policy_dict = {policy1: "int8_32"}
    assert policy_dict[policy2] == "int8_32"

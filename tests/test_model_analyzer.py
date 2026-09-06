"""Tests for precision-aware model analysis."""

import pytest
from unittest.mock import Mock, patch
from sparse_llm.loading.model_analyzer import ModelAnalyzer, ModelAnalysis
from sparse_llm.loading.model_introspector import ModelInfo
from sparse_llm.loading.precision_calculator import Precision


class TestModelAnalyzer:
    """Test ModelAnalyzer with precision awareness."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for tests."""
        return ModelAnalyzer()

    @pytest.fixture
    def mock_mixtral_info(self):
        """Mock ModelInfo for Mixtral 8x7B at FP16."""
        return ModelInfo(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            is_moe=True,
            num_layers=32,
            num_experts=8,
            num_experts_per_tok=2,
            shared_weight_bytes=3_000_000_000,  # ~3GB at FP16
            expert_weight_bytes=437_000_000,  # ~437MB per expert at FP16
            total_bytes=115_000_000_000,  # ~115GB at FP16 (3GB + 8*32*437MB)
        )

    def test_analyze_from_model_info_fp16(self, analyzer, mock_mixtral_info):
        """Test analysis at FP16 (baseline)."""
        analysis = analyzer.analyze_from_model_info(mock_mixtral_info, Precision.FP16)

        assert analysis.precision == Precision.FP16
        assert analysis.shared_weight_bytes == 3_000_000_000
        assert analysis.expert_weight_bytes == 437_000_000
        assert analysis.total_bytes == 115_000_000_000

        # Per-token active: shared + 2 experts * 32 layers
        expected_active = 3_000_000_000 + (2 * 437_000_000 * 32)
        assert analysis.per_token_active_bytes == expected_active

    def test_analyze_from_model_info_fp32(self, analyzer, mock_mixtral_info):
        """Test analysis at FP32 (2x FP16)."""
        analysis = analyzer.analyze_from_model_info(mock_mixtral_info, Precision.FP32)

        assert analysis.precision == Precision.FP32
        # Should be 2x the FP16 values
        assert analysis.shared_weight_bytes == 6_000_000_000  # 2x 3GB
        assert analysis.expert_weight_bytes == 874_000_000  # 2x 437MB
        assert analysis.total_bytes == 230_000_000_000  # 2x 115GB

    def test_analyze_from_model_info_int8(self, analyzer, mock_mixtral_info):
        """Test analysis at INT8 (0.5x FP16)."""
        analysis = analyzer.analyze_from_model_info(mock_mixtral_info, Precision.INT8)

        assert analysis.precision == Precision.INT8
        # Should be 0.5x the FP16 values
        assert analysis.shared_weight_bytes == 1_500_000_000  # 0.5x 3GB
        assert analysis.expert_weight_bytes == 218_500_000  # 0.5x 437MB
        assert analysis.total_bytes == 57_500_000_000  # 0.5x 115GB

    def test_analyze_from_model_info_int4(self, analyzer, mock_mixtral_info):
        """Test analysis at INT4 (0.25x FP16)."""
        analysis = analyzer.analyze_from_model_info(mock_mixtral_info, Precision.INT4)

        assert analysis.precision == Precision.INT4
        # Should be 0.25x the FP16 values
        assert analysis.shared_weight_bytes == 750_000_000  # 0.25x 3GB
        assert analysis.expert_weight_bytes == 109_250_000  # 0.25x 437MB
        assert analysis.total_bytes == 28_750_000_000  # 0.25x 115GB

    def test_analyze_from_model_info_per_token_active_moe(self, analyzer, mock_mixtral_info):
        """Test per-token active calculation for MoE."""
        analysis = analyzer.analyze_from_model_info(mock_mixtral_info, Precision.FP16)

        # Per-token active = shared + (num_experts_per_tok * expert_bytes * num_layers)
        expected = 3_000_000_000 + (2 * 437_000_000 * 32)
        assert analysis.per_token_active_bytes == expected

        # Should be much less than total (only 2 of 8 experts active)
        assert analysis.per_token_active_bytes < analysis.total_bytes

    def test_analyze_from_model_info_dense_model(self, analyzer):
        """Test analysis for dense (non-MoE) model."""
        dense_info = ModelInfo(
            model_id="meta-llama/Llama-2-7b-hf",
            is_moe=False,
            num_layers=32,
            num_experts=0,
            num_experts_per_tok=0,
            shared_weight_bytes=14_000_000_000,  # 14GB at FP16
            expert_weight_bytes=0,
            total_bytes=14_000_000_000,
        )

        analysis = analyzer.analyze_from_model_info(dense_info, Precision.FP16)

        assert analysis.precision == Precision.FP16
        assert analysis.shared_weight_bytes == 14_000_000_000
        assert analysis.expert_weight_bytes == 0
        assert analysis.total_bytes == 14_000_000_000

        # Dense model: all weights active per token
        assert analysis.per_token_active_bytes == analysis.total_bytes

    def test_compare_precisions(self, analyzer, mock_mixtral_info):
        """Test comparing all precisions."""
        with patch.object(analyzer.introspector, 'introspect', return_value=mock_mixtral_info):
            analyses = analyzer.compare_precisions("mistralai/Mixtral-8x7B-Instruct-v0.1")

        # Should have all four precisions
        assert Precision.FP32 in analyses
        assert Precision.FP16 in analyses
        assert Precision.INT8 in analyses
        assert Precision.INT4 in analyses

        # Check relative sizes
        fp32_bytes = analyses[Precision.FP32].total_bytes
        fp16_bytes = analyses[Precision.FP16].total_bytes
        int8_bytes = analyses[Precision.INT8].total_bytes
        int4_bytes = analyses[Precision.INT4].total_bytes

        # FP32 should be 2x FP16
        assert fp32_bytes == 2 * fp16_bytes
        # INT8 should be 0.5x FP16
        assert int8_bytes * 2 == fp16_bytes
        # INT4 should be 0.25x FP16
        assert int4_bytes * 4 == fp16_bytes

    def test_property_total_gb(self, analyzer, mock_mixtral_info):
        """Test total_gb property."""
        analysis = analyzer.analyze_from_model_info(mock_mixtral_info, Precision.FP16)

        expected_gb = 115_000_000_000 / (1024 ** 3)
        assert abs(analysis.total_gb - expected_gb) < 0.01

    def test_property_shared_weight_gb(self, analyzer, mock_mixtral_info):
        """Test shared_weight_gb property."""
        analysis = analyzer.analyze_from_model_info(mock_mixtral_info, Precision.FP16)

        expected_gb = 3_000_000_000 / (1024 ** 3)
        assert abs(analysis.shared_weight_gb - expected_gb) < 0.01

    def test_property_expert_weight_mb(self, analyzer, mock_mixtral_info):
        """Test expert_weight_mb property."""
        analysis = analyzer.analyze_from_model_info(mock_mixtral_info, Precision.FP16)

        expected_mb = 437_000_000 / (1024 ** 2)
        assert abs(analysis.expert_weight_mb - expected_mb) < 0.1

    def test_property_per_token_active_gb(self, analyzer, mock_mixtral_info):
        """Test per_token_active_gb property."""
        analysis = analyzer.analyze_from_model_info(mock_mixtral_info, Precision.FP16)

        expected_bytes = 3_000_000_000 + (2 * 437_000_000 * 32)
        expected_gb = expected_bytes / (1024 ** 3)
        assert abs(analysis.per_token_active_gb - expected_gb) < 0.01

    def test_analyze_with_string_precision(self, analyzer, mock_mixtral_info):
        """Test that string precision is parsed correctly."""
        with patch.object(analyzer.introspector, 'introspect', return_value=mock_mixtral_info):
            # Should accept string precision
            analysis = analyzer.analyze("model-id", precision="fp32")
            assert analysis.precision == Precision.FP32

            analysis = analyzer.analyze("model-id", precision="int8")
            assert analysis.precision == Precision.INT8

    def test_analyze_from_config_with_string_precision(self, analyzer, mock_mixtral_info):
        """Test analyze_from_config with string precision."""
        mock_config = Mock()

        with patch.object(analyzer.introspector, 'introspect_from_config', return_value=mock_mixtral_info):
            analysis = analyzer.analyze_from_config(mock_config, "model-id", precision="int4")
            assert analysis.precision == Precision.INT4


class TestModelAnalysisDataclass:
    """Test ModelAnalysis dataclass properties."""

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample ModelAnalysis for testing."""
        model_info = ModelInfo(
            model_id="test-model",
            is_moe=True,
            num_layers=32,
            num_experts=8,
            num_experts_per_tok=2,
            shared_weight_bytes=3_000_000_000,
            expert_weight_bytes=437_000_000,
            total_bytes=115_000_000_000,
        )

        return ModelAnalysis(
            model_info=model_info,
            precision=Precision.FP16,
            shared_weight_bytes=3_000_000_000,
            expert_weight_bytes=437_000_000,
            total_bytes=115_000_000_000,
            per_token_active_bytes=31_000_000_000,
        )

    def test_dataclass_immutable(self, sample_analysis):
        """Test that ModelAnalysis is frozen (immutable)."""
        with pytest.raises(AttributeError):
            sample_analysis.total_bytes = 999

    def test_contains_model_info(self, sample_analysis):
        """Test that analysis contains original ModelInfo."""
        assert sample_analysis.model_info.model_id == "test-model"
        assert sample_analysis.model_info.is_moe is True
        assert sample_analysis.model_info.num_experts == 8

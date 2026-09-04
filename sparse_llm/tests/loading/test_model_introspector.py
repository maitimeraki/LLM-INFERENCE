"""Tests for ModelIntrospector."""

from unittest.mock import Mock
import pytest


def test_detect_moe_model_mixtral():
    from sparse_llm.loading.model_introspector import ModelIntrospector

    # Mock Mixtral config
    config = Mock()
    config.num_hidden_layers = 32
    config.num_local_experts = 8
    config.num_experts_per_tok = 2
    config.hidden_size = 4096
    config.intermediate_size = 14336
    config.vocab_size = 32000

    introspector = ModelIntrospector()
    model_info = introspector.introspect_from_config(config, model_id="mixtral-test")

    assert model_info.is_moe == True
    assert model_info.num_experts == 8
    assert model_info.num_experts_per_tok == 2
    assert model_info.expert_weight_bytes > 0


def test_detect_dense_model_llama():
    from sparse_llm.loading.model_introspector import ModelIntrospector

    # Mock Llama config (dense model, no MoE)
    config = Mock()
    config.num_hidden_layers = 32
    config.hidden_size = 4096
    config.intermediate_size = 11008
    config.vocab_size = 32000
    # No num_local_experts attribute

    introspector = ModelIntrospector()
    model_info = introspector.introspect_from_config(config, model_id="llama-test")

    assert model_info.is_moe == False
    assert model_info.num_experts == 0
    assert model_info.expert_weight_bytes == 0

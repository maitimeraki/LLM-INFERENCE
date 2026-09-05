"""Tests for the safe generic adapter used for unknown MoE models."""

from types import SimpleNamespace

import torch

from sparse_llm import DevicePolicy, UniversalMoEAdapter


class FakeTokenizer:
    eos_token_id = None

    def __call__(self, prompt, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        return " ".join(str(token) for token in token_ids)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            model_type="unknown_moe",
            architectures=["UnknownMoeForCausalLM"],
            num_hidden_layers=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            use_cache=True,
        )

    def forward(self, input_ids, **kwargs):
        logits = torch.full((1, input_ids.shape[1], 8), -100.0)
        logits[:, -1, (int(input_ids[:, -1].item()) + 1) % 8] = 100.0
        return {"logits": logits, "past_key_values": None}


def test_unknown_moe_uses_generic_generation_path():
    model = FakeModel()
    adapter = UniversalMoEAdapter(
        "local/unknown-moe",
        policy=DevicePolicy(device="cpu"),
        tokenizer=FakeTokenizer(),
        model=model,
        config=model.config,
    )

    result = adapter.generate("ignored", max_new_tokens=2)

    assert result.token_ids == [1, 2, 3, 4]
    assert result.metrics.generated_tokens == 2
    assert adapter.capabilities.is_moe
    assert adapter.capabilities.expert_paging is False


def test_unvalidated_moe_paging_is_unavailable():
    adapter = UniversalMoEAdapter("local/unknown-moe", policy=DevicePolicy(device="cpu"))
    adapter._capabilities = adapter._capabilities.__class__(
        model_id="local/unknown-moe",
        model_type="unknown_moe",
        is_moe=True,
        num_hidden_layers=1,
        num_experts=2,
        top_k_experts=1,
    )

    validation = adapter.validate_paging()

    assert not validation.eligible
    assert "architecture-validated" in validation.failure_reasons[0]
    assert adapter.paging_capabilities is None


def test_compatibility_adapter_has_no_guessed_paging_execution():
    adapter = UniversalMoEAdapter("local/unknown-moe")

    assert not hasattr(adapter, "_create_tensor_mapper")
    assert not hasattr(adapter, "_fallback_expert_forward")
    assert not hasattr(adapter, "_generate_with_paging")

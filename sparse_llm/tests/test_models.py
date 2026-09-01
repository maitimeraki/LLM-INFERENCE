from types import SimpleNamespace

import torch

from sparse_llm import (
    DevicePolicy,
    InferenceEngine,
    ModelRegistry,
    TransformersCausalLMAdapter,
)


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
            model_type="fake",
            architectures=["FakeForCausalLM"],
            num_hidden_layers=2,
            use_cache=True,
        )
        self.calls = []

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True, **kwargs):
        self.calls.append((input_ids.clone(), past_key_values))
        vocab = 8
        next_id = (int(input_ids[:, -1].item()) + 1) % vocab
        logits = torch.full((1, input_ids.shape[1], vocab), -100.0)
        logits[:, -1, next_id] = 100.0
        return {"logits": logits, "past_key_values": None}


def test_fake_model_generation_is_real_and_deterministic():
    model = FakeModel()
    adapter = TransformersCausalLMAdapter(
        "local/fake",
        policy=DevicePolicy(device="cpu"),
        tokenizer=FakeTokenizer(),
        model=model,
        config=model.config,
    )

    result = adapter.generate("ignored", max_new_tokens=3)

    assert result.token_ids == [1, 2, 3, 4, 5]
    assert result.text == "1 2 3 4 5"
    assert result.metrics.prompt_tokens == 2
    assert result.metrics.generated_tokens == 3
    assert result.metrics.prefill_latency_ms >= 0
    assert result.metrics.decode_latency_ms >= 0
    assert result.metrics.cache_hits == 0
    assert len(model.calls) == 4


def test_moe_capability_is_reported_without_paging_claim():
    model = FakeModel()
    model.config.num_local_experts = 8
    model.config.num_experts_per_tok = 2
    adapter = TransformersCausalLMAdapter(
        "local/moe",
        policy=DevicePolicy(device="cpu"),
        tokenizer=FakeTokenizer(),
        model=model,
        config=model.config,
    )

    capabilities = adapter.capabilities

    assert capabilities.is_moe
    assert capabilities.num_experts == 8
    assert capabilities.top_k_experts == 2
    assert not capabilities.expert_paging
    assert capabilities.to_dict()["architecture_classification"] == "moe"


def test_default_registry_uses_generic_adapter_without_name_heuristics():
    from sparse_llm import create_model_adapter, get_default_registry

    adapter = create_model_adapter("mistralai/Mixtral-8x7B-v0.1", policy=DevicePolicy(device="cpu"))

    assert isinstance(adapter, TransformersCausalLMAdapter)
    assert adapter.capabilities.expert_paging is False
    assert "universal-moe" not in get_default_registry().names()


def test_registry_specializations_require_checkpoint_config():
    registry = ModelRegistry()
    selected = []

    def factory(model_id, *, policy, config=None, **kwargs):
        selected.append(config)
        return TransformersCausalLMAdapter(model_id, policy=policy, config=config)

    registry.register(
        "validated-specialization",
        factory,
        lambda config: getattr(config, "model_type", None) == "validated",
        priority=10,
    )

    generic = registry.create("org/validated", policy=DevicePolicy(device="cpu"))
    specialized = registry.create(
        "org/validated",
        policy=DevicePolicy(device="cpu"),
        config=SimpleNamespace(model_type="validated"),
    )

    assert isinstance(generic, TransformersCausalLMAdapter)
    assert specialized is not generic
    assert selected == [SimpleNamespace(model_type="validated")]


def test_engine_delegates_to_adapter():
    model = FakeModel()
    adapter = TransformersCausalLMAdapter(
        "local/fake",
        policy=DevicePolicy(device="cpu"),
        tokenizer=FakeTokenizer(),
        model=model,
        config=model.config,
    )

    result = InferenceEngine(adapter=adapter).generate("prompt", max_new_tokens=1)

    assert result.metrics.generated_tokens == 1


def test_engine_uses_process_wide_default_registry(monkeypatch):
    from sparse_llm.inference import engine as engine_module

    calls = []
    expected = TransformersCausalLMAdapter("local/fake", policy=DevicePolicy(device="cpu"))

    class Registry:
        def create(self, model_id, *, policy):
            calls.append((model_id, policy))
            return expected

    monkeypatch.setattr(engine_module, "get_default_registry", lambda: Registry(), raising=False)
    monkeypatch.setattr(
        engine_module,
        "ModelRegistry",
        lambda: (_ for _ in ()).throw(AssertionError("fresh registry was created")),
    )

    engine = InferenceEngine(model="org/model", device="cpu")

    assert engine.adapter is expected
    assert calls[0][0] == "org/model"


def test_loading_options_are_forwarded_lazily():
    calls = []

    class Loader:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((cls.__name__, model_id, kwargs))
            if cls.__name__ == "TokenizerLoader":
                return FakeTokenizer()
            return FakeModel()

    class TokenizerLoader(Loader):
        pass

    class ModelLoader(Loader):
        pass

    adapter = TransformersCausalLMAdapter(
        "org/model",
        policy=DevicePolicy(
            device="cpu",
            dtype="float32",
            revision="rev-1",
            local_files_only=True,
            trust_remote_code=False,
        ),
        tokenizer_loader=TokenizerLoader,
        model_loader=ModelLoader,
    )

    assert calls == []
    adapter.load()
    assert len(calls) == 2
    assert calls[0][2]["revision"] == "rev-1"
    assert calls[0][2]["local_files_only"] is True
    assert calls[1][2]["torch_dtype"] is torch.float32


def test_device_policy_rejects_invalid_device():
    try:
        DevicePolicy(device="not-a-device").resolve_device()
    except ValueError as error:
        assert "unsupported device" in str(error)
    else:
        raise AssertionError("invalid device was accepted")

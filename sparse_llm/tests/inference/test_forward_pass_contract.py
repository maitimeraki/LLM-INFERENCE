"""Forward-pass correctness contract for the custom PyTorch MoE engine.

Encodes the six invariants from
`.superpowers/sdd/OPTIMIZATION_SUMMARY/task-1-brief.md`. Every test is expected
to FAIL on the current (pre-fix) code and PASS after Tasks 2/3.

The full `CustomMoEInferenceEngine` is deliberately never constructed here: its
`__init__` downloads a tokenizer, introspects a model, and allocates real
memory. Where a defect lives in the engine's orchestration loop we assert on
the post-fix interface / on the per-layer units instead (see the task report
for which assertion uses which strategy).

Fixed synthetic configuration (verbatim from the brief):
    num_layers = 3, num_experts = 4, top_k = 2, hidden_size = 64,
    num_attention_heads = 4 (head_dim = 16), intermediate_size = 128,
    max_seq_len = 32, device = cpu, dtype = float32.
"""

import inspect

import pytest
import torch

from sparse_llm.inference.attention_engine import AttentionEngine

# --- fixed synthetic config (verbatim from the brief) ---
NUM_LAYERS = 3
NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = 16
INTERMEDIATE_SIZE = 128
MAX_SEQ_LEN = 32
DEVICE = "cpu"
DTYPE = torch.float32

torch.manual_seed(0)


def _make_attention_engine():
    return AttentionEngine(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        device=DEVICE,
        dtype=DTYPE,
    )


def _rms(x):
    return x.pow(2).mean(dim=-1, keepdim=True).sqrt()


# ---------------------------------------------------------------------------
# Assertion 1 - all layers visited
# ---------------------------------------------------------------------------
def test_all_layers_visited_in_decode():
    """Assertion 1: one decode step must enter every layer.

    Defect: `_decode_phase_streaming` pins `layer_id = 0`, so the streaming
    path only ever visits layer 0. Strategy: drive the per-layer unit in a
    loop (the engine's loop after Task 2) and assert every layer is entered;
    also assert the streaming path no longer hardcodes layer 0.
    """
    engine = _make_attention_engine()
    _, cache = engine.prefill(torch.randn(1, 4, engine.hidden_dim))

    visited = []
    real_decode = engine.decode

    def spy(token, kv, layer_idx=None, **kwargs):
        visited.append(layer_idx)
        return real_decode(token, kv, layer_idx=layer_idx, **kwargs)

    engine.decode = spy

    hidden = torch.randn(1, 1, engine.hidden_dim)
    for layer_idx in range(engine.num_layers):
        hidden, cache = engine.decode(hidden, cache, layer_idx=layer_idx)

    assert set(visited) == set(range(engine.num_layers)), (
        f"decode visited layers {sorted(set(visited))}, expected all "
        f"{engine.num_layers} layers"
    )


def test_streaming_decode_does_not_pin_layer_zero():
    """Assertion 1 (source-level): the streaming path must not hardcode layer 0."""
    from sparse_llm.inference.moe_inference_engine import CustomMoEInferenceEngine

    src = inspect.getsource(CustomMoEInferenceEngine._decode_phase_streaming)
    assert "layer_id = 0" not in src, (
        "streaming decode hardcodes `layer_id = 0`; it must run every layer."
    )


# ---------------------------------------------------------------------------
# Assertion 2 - attention uses learned projections
# ---------------------------------------------------------------------------
def _projection_checkpoint(scale_q):
    """Checkpoint-style weights with distinct attention projections per layer."""
    eye = torch.eye(HIDDEN_SIZE)
    weights = {}
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        weights[f"{prefix}.q_proj.weight"] = eye * scale_q
        weights[f"{prefix}.k_proj.weight"] = eye
        weights[f"{prefix}.v_proj.weight"] = eye
        weights[f"{prefix}.o_proj.weight"] = eye
    return weights


def test_attention_uses_learned_projections():
    """Assertion 2: differing q_proj weights must change the layer output.

    Defect: the input is reshaped three times for Q/K/V, so projection weights
    are invisible. Two engines are given different q_proj weights; their
    outputs must differ by more than 1e-6.
    """
    x = torch.randn(1, 4, HIDDEN_SIZE)

    engine_a = _make_attention_engine()
    engine_a.load_weights_from_checkpoint(_projection_checkpoint(scale_q=0.5))
    out_a, _ = engine_a.prefill(x.clone())

    engine_b = _make_attention_engine()
    engine_b.load_weights_from_checkpoint(_projection_checkpoint(scale_q=2.0))
    out_b, _ = engine_b.prefill(x.clone())

    assert torch.max(torch.abs(out_a - out_b)).item() > 1e-6, (
        "Changing q_proj weights did not change the attention output - the "
        "engine is not using learned Q/K/V/O projections."
    )


# ---------------------------------------------------------------------------
# Assertion 3 - RMSNorm normalizes
# ---------------------------------------------------------------------------
def test_rmsnorm_normalizes():
    """Assertion 3: RMSNorm output has unit RMS; per-channel weight scales it."""
    try:
        from sparse_llm.inference.attention_engine import RMSNorm
    except ImportError:
        from sparse_llm.inference.moe_inference_engine import RMSNorm

    x = torch.randn(2, 8, HIDDEN_SIZE) * 3.0 + 1.5
    norm = RMSNorm(HIDDEN_SIZE)

    with torch.no_grad():
        norm.weight.fill_(1.0)
    y = norm(x)
    got = _rms(y).squeeze(-1)
    assert torch.allclose(got, torch.ones_like(got), atol=1e-3), (
        f"RMSNorm output RMS is {got.flatten()[:4].tolist()}, expected ~1.0"
    )

    with torch.no_grad():
        norm.weight.fill_(2.0)
    y2 = norm(x)
    assert torch.allclose(y2, 2.0 * y, atol=1e-4), (
        "Scaling the RMSNorm weight must scale the normalized output per channel."
    )


# ---------------------------------------------------------------------------
# Assertion 4 - residual algebra
# ---------------------------------------------------------------------------
def test_residual_algebra_zeroed_sublayer_is_identity():
    """Assertion 4: block output == input when the sublayer output is zeroed.

    Defect: the engine replaces the hidden state instead of adding
    (`h + 0 != h`). Strategy: exercise the per-layer attention block on a bare
    instance (no `__init__`) with a zero-output attention sublayer.
    """
    from sparse_llm.inference.moe_inference_engine import CustomMoEInferenceEngine

    method_name = None
    for name in ("_apply_attention_layer", "_apply_attention_block", "_attention_block"):
        if hasattr(CustomMoEInferenceEngine, name):
            method_name = name
            break
    assert method_name is not None, (
        "engine must expose a per-layer attention block that adds a residual "
        "(h + sublayer(h)); current code replaces the hidden state."
    )

    class _ZeroAttention:
        def decode(self, *args, **kwargs):
            return torch.zeros(1, 1, HIDDEN_SIZE), kwargs.get("kv_cache")

    engine = object.__new__(CustomMoEInferenceEngine)
    engine.num_layers = NUM_LAYERS
    engine.attention_engine = _ZeroAttention()
    engine.device = torch.device(DEVICE)
    engine.dtype = DTYPE
    engine.attention_weights = {}
    engine.layer_norms = {
        li: {
            "input_layernorm": torch.ones(HIDDEN_SIZE),
            "post_attention_layernorm": torch.ones(HIDDEN_SIZE),
        }
        for li in range(NUM_LAYERS)
    }

    h = torch.randn(1, 1, HIDDEN_SIZE)
    out, _ = getattr(engine, method_name)(h, 0, {})
    assert torch.equal(out, h), (
        "With a zeroed attention sublayer the block must return its input "
        "(residual add), but it did not."
    )


# ---------------------------------------------------------------------------
# Assertion 5 - KV cache position advances once per token
# ---------------------------------------------------------------------------
def test_kv_cache_position_advances_once_per_token():
    """Assertion 5: seq_len == P + N after prefill of P and N decode steps.

    Defect: the engine's per-layer loop calls decode once per layer, and decode
    advances seq_len on every call, so the position advances once per inner
    layer instead of once per token.
    """
    engine = _make_attention_engine()
    prompt_len = 5
    _, cache = engine.prefill(torch.randn(1, prompt_len, engine.hidden_dim))

    decode_steps = 3
    hidden = torch.randn(1, 1, engine.hidden_dim)
    for _ in range(decode_steps):
        for layer_idx in range(engine.num_layers):
            hidden, cache = engine.decode(hidden, cache, layer_idx=layer_idx)

    assert cache["seq_len"] == prompt_len + decode_steps, (
        f"KV cache seq_len is {cache['seq_len']}, expected "
        f"{prompt_len + decode_steps} ({prompt_len} prefill + {decode_steps} "
        f"decode tokens). The position must advance once per token, not once "
        f"per layer."
    )


# ---------------------------------------------------------------------------
# Assertion 6 - layer loop lives in exactly one place
# ---------------------------------------------------------------------------
def test_layer_loop_lives_in_exactly_one_place():
    """Assertion 6: prefill/decode take layer_idx and do not loop layers internally."""
    for name in ("prefill", "decode"):
        method = getattr(AttentionEngine, name)
        params = inspect.signature(method).parameters
        assert "layer_idx" in params, (
            f"AttentionEngine.{name} must accept a layer_idx argument (the "
            f"layer loop belongs to the engine); got params {list(params)}"
        )

    src = inspect.getsource(AttentionEngine.prefill) + inspect.getsource(AttentionEngine.decode)
    assert "range(self.num_layers)" not in src, (
        "AttentionEngine.prefill/decode must not iterate range(self.num_layers); "
        "the layer loop must live in exactly one place (the engine)."
    )
"""Forward-pass correctness contract for the custom PyTorch MoE engine.

Encodes the six invariants from
`.superpowers/sdd/OPTIMIZATION_SUMMARY/task-1-brief.md`. Every test is expected
to FAIL on the current (pre-fix) code and PASS after Tasks 2/3.

The full `CustomMoEInferenceEngine` is never constructed via `__init__` (it
downloads a tokenizer, introspects a model, and allocates real memory). Where a
defect lives in the engine's orchestration loop, the bare engine is built with
`object.__new__` and its collaborators stubbed, then the real engine loop is
driven and observed.

API names pinned by Ruling R7 (harness and Task 2 must not drift):
    AttentionEngine.load_layer_weights(shared_weights, layer_id)
    AttentionEngine.prefill(input_ids, layer_idx, kv_cache)
    AttentionEngine.decode(token_id, layer_idx, kv_cache)
    CustomMoEInferenceEngine._apply_attention_layer(h, layer_idx, kv_cache)
        -> (h, kv_cache)
    RMSNorm lives in sparse_llm/inference/attention_engine.py, .weight [hidden]

Fixed synthetic configuration (verbatim from the brief):
    num_layers = 3, num_experts = 4, top_k = 2, hidden_size = 64,
    num_attention_heads = 4 (head_dim = 16), max_seq_len = 32,
    device = cpu, dtype = float32.
"""

import inspect
from types import SimpleNamespace

import torch

from sparse_llm.inference.attention_engine import AttentionEngine

# --- fixed synthetic config (verbatim from the brief) ---
NUM_LAYERS = 3
NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = 16
MAX_SEQ_LEN = 32
DEVICE = "cpu"
DTYPE = torch.float32

# INTERMEDIATE_SIZE (128) is intentionally omitted: no expert FFN is exercised
# by these contract tests, and a dead constant reads as an unused fixture.

assert TOP_K <= NUM_EXPERTS, "synthetic config: top_k must not exceed num_experts"

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
# Bare-engine harness: drives the real orchestration loop with stubbed units.
# ---------------------------------------------------------------------------
class _FixedRouter:
    """Returns valid top-k routing; never touches real weights."""

    router_weights = None

    def forward(self, hidden_states, batch_mode=True):
        batch = hidden_states.shape[0]
        indices = torch.arange(TOP_K).view(1, 1, TOP_K).expand(batch, 1, TOP_K).clone()
        weights = torch.ones(batch, 1, TOP_K, dtype=torch.float32)
        return indices, weights


class _PassthroughExperts:
    def process_single(self, hidden_state, **kwargs):
        return hidden_state


def _make_bare_engine(attention_engine):
    from sparse_llm.inference.moe_inference_engine import CustomMoEInferenceEngine

    engine = object.__new__(CustomMoEInferenceEngine)
    engine.model_info = SimpleNamespace(num_layers=NUM_LAYERS)
    engine.attention_engine = attention_engine
    engine.router_calculator = _FixedRouter()
    engine.expert_processor = _PassthroughExperts()
    engine.tokenizer = SimpleNamespace(eos_token_id=999)
    engine._sample_next_token = lambda *a, **k: torch.tensor([0])
    engine._embed_tokens = lambda *a, **k: torch.zeros(1, 1, HIDDEN_SIZE)
    engine._create_expert_loader = lambda layer_id: (lambda expert_id: None)
    return engine


def _make_state(kv_cache):
    return SimpleNamespace(
        hidden_states=torch.zeros(1, 1, HIDDEN_SIZE),
        kv_cache=kv_cache,
        token_times=[],
    )


# ---------------------------------------------------------------------------
# Assertion 1 - all layers visited
# ---------------------------------------------------------------------------
class _LayerSpy:
    """Records the layer_idx the engine passes to `decode` on every call."""

    def __init__(self):
        self.visited = []

    def decode(self, *args, **kwargs):
        if "layer_idx" in kwargs:
            layer_idx = kwargs["layer_idx"]
        elif len(args) >= 2:
            layer_idx = args[1]
        else:
            layer_idx = None
        self.visited.append(layer_idx)
        token = args[0] if args else kwargs.get("token_id")
        cache = kwargs.get("kv_cache")
        if cache is None and len(args) >= 3:
            cache = args[2]
        return token, cache


def test_engine_decode_loop_enters_every_layer():
    """Assertion 1: the engine's per-token decode loop must enter every layer.

    Drives the real `_decode_phase` with a spy attention sublayer and asserts
    the engine passed every `layer_idx` in `range(num_layers)` to `decode`.
    Current code calls `decode` without a `layer_idx` at all, so the spy sees
    `{None}` and the test fails for the real reason (not by construction).
    """
    spy = _LayerSpy()
    engine = _make_bare_engine(spy)
    state = _make_state({})

    engine._decode_phase(state, max_tokens=1, temperature=1.0, top_p=1.0, top_k=1)

    assert set(spy.visited) == set(range(NUM_LAYERS)), (
        f"engine decode passed layers {sorted(map(str, set(spy.visited)))}, "
        f"expected all {NUM_LAYERS} layers"
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
def _checkpoint(q_scale, k_scale, v_scale, o_scale):
    """Checkpoint-style weights with distinct projections on every path."""
    weights = {}
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        weights[f"{prefix}.q_proj.weight"] = torch.eye(HIDDEN_SIZE) * q_scale
        weights[f"{prefix}.k_proj.weight"] = torch.eye(HIDDEN_SIZE) * k_scale
        weights[f"{prefix}.v_proj.weight"] = torch.eye(HIDDEN_SIZE) * v_scale
        weights[f"{prefix}.o_proj.weight"] = torch.eye(HIDDEN_SIZE) * o_scale
    return weights


def test_attention_uses_learned_projections():
    """Assertion 2: differing q/k/v/o weights must change the layer output.

    Defect: the input is reshaped three times for Q/K/V, so projection weights
    are invisible. Two engines get different q, k, v, AND o weights; their
    outputs must differ by more than 1e-6.
    """
    x = torch.randn(1, 4, HIDDEN_SIZE)

    engine_a = _make_attention_engine()
    engine_a.load_layer_weights(_checkpoint(0.5, 0.7, 1.3, 0.9), layer_id=0)
    out_a, _ = engine_a.prefill(input_ids=x.clone(), layer_idx=0, kv_cache=None)

    engine_b = _make_attention_engine()
    engine_b.load_layer_weights(_checkpoint(2.0, 1.1, 0.6, 1.4), layer_id=0)
    out_b, _ = engine_b.prefill(input_ids=x.clone(), layer_idx=0, kv_cache=None)

    assert torch.max(torch.abs(out_a - out_b)).item() > 1e-6, (
        "Changing attention projection weights did not change the attention "
        "output - the engine is not using learned Q/K/V/O projections."
    )


# ---------------------------------------------------------------------------
# Assertion 3 - RMSNorm normalizes
# ---------------------------------------------------------------------------
def test_rmsnorm_normalizes():
    """Assertion 3: RMSNorm output has unit RMS; per-channel weight scales it."""
    from sparse_llm.inference.attention_engine import RMSNorm

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
class _ZeroAttention:
    """Attention sublayer that emits zeros, keeping the residual observable."""

    def decode(self, *args, **kwargs):
        token = args[0] if args else kwargs.get("token_id")
        cache = kwargs.get("kv_cache")
        if cache is None and len(args) >= 3:
            cache = args[2]
        return torch.zeros_like(token), cache


def test_residual_algebra_zeroed_sublayer_is_identity():
    """Assertion 4: block output == input when the sublayer output is zeroed.

    Defect: the engine replaces the hidden state instead of adding
    (`h + 0 != h`). Uses the R7-pinned block
    `_apply_attention_layer(h, layer_idx, kv_cache) -> (h, kv_cache)`.
    """
    from sparse_llm.inference.moe_inference_engine import CustomMoEInferenceEngine

    assert hasattr(CustomMoEInferenceEngine, "_apply_attention_layer"), (
        "engine must expose `_apply_attention_layer(h, layer_idx, kv_cache)` "
        "that adds the attention output as a residual (h + sublayer(h)); "
        "current code replaces the hidden state."
    )

    engine = object.__new__(CustomMoEInferenceEngine)
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
    out, _ = engine._apply_attention_layer(h, 0, {})
    assert torch.equal(out, h), (
        "With a zeroed attention sublayer the block must return its input "
        "(residual add), but it did not."
    )


# ---------------------------------------------------------------------------
# Assertion 5 - KV cache position advances once per token
# ---------------------------------------------------------------------------
def test_kv_cache_position_advances_once_per_token():
    """Assertion 5 contract: `AttentionEngine.decode` writes K/V at
    `current_pos` and does NOT advance `kv_cache['seq_len']`; the engine
    advances `seq_len` exactly once per generated token, after its layer loop.

    Defect pinned: current `decode` increments `seq_len` on every per-layer
    call, so one token traversing `num_layers` layers advances the position
    `num_layers` times instead of once.
    """
    engine = _make_bare_engine(_make_attention_engine())
    prompt_len, decode_steps = 5, 3

    # Post-prefill cache state: P tokens already written, position == P.
    cache = {
        "keys": torch.zeros(1, NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM),
        "values": torch.zeros(1, NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM),
        "seq_len": prompt_len,
    }
    state = _make_state(cache)

    engine._decode_phase(
        state, max_tokens=decode_steps, temperature=1.0, top_p=1.0, top_k=1
    )

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
        assert "kv_cache" in params, (
            f"AttentionEngine.{name} must accept a kv_cache argument; "
            f"got params {list(params)}"
        )

    src = inspect.getsource(AttentionEngine.prefill) + inspect.getsource(
        AttentionEngine.decode
    )
    assert "range(self.num_layers)" not in src, (
        "AttentionEngine.prefill/decode must not iterate range(self.num_layers); "
        "the layer loop must live in exactly one place (the engine)."
    )
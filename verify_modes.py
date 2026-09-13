"""Verification script for tiled/fused processing modes."""
from sparse_llm.inference.moe_inference_engine import InferenceConfig
from sparse_llm.inference.expert_processor import ExpertProcessor
from sparse_llm.loading.expert_cache import ExpertCache

print("=" * 60)
print("TEST 1: InferenceConfig mode parameter")
print("=" * 60)
for mode in ['standard', 'tiled', 'fused']:
    c = InferenceConfig(model_path='test', expert_processing_mode=mode)
    print(f"  {mode}: {c.expert_processing_mode}")

print()
print("=" * 60)
print("TEST 2: ExpertProcessor accepts processing_mode")
print("=" * 60)
cache = ExpertCache(gpu_slots=8, cpu_slots=8, expert_bytes=1000000)
for mode in ['standard', 'tiled', 'fused']:
    proc = ExpertProcessor(
        expert_cache=cache,
        num_experts=8,
        hidden_dim=4096,
        expert_dim=11008,
        processing_mode=mode
    )
    print(f"  {mode}: {proc.processing_mode}")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)

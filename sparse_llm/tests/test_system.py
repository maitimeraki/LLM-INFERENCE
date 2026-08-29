from sparse_llm import InferenceConfig, SparseInference
import logging
logging.basicConfig(level=logging.WARNING)
print("\nInitializing full SparseLLM system...")
config = InferenceConfig(
    model_name='gpt2',
    vram_target_gb=6,
    enable_prediction=True,
    enable_prefetch=True,
    enable_quantization=True,
    enable_production=True
)
inference = SparseInference(config)
print("System initialized successfully")
active = {k: v for k, v in inference.phase_levels.items() if v}
print(f"Active phases: {list(active.keys())}")
stats = inference.get_stats()
print(f"Stats modules: {list(stats.keys())}")
print("Production-grade SparseLLM system ready!")

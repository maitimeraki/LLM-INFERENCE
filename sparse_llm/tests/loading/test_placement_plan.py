"""Tests for PlacementPlan dataclass."""


def test_placement_plan_creation():
    from sparse_llm.loading.placement_plan import PlacementPlan
    from sparse_llm.loading.model_introspector import ModelInfo

    model_info = ModelInfo(
        model_id="test-model",
        is_moe=True,
        num_layers=32,
        num_experts=8,
        num_experts_per_tok=2,
        shared_weight_bytes=2*1024**3,
        expert_weight_bytes=180*1024**2,
        total_bytes=50*1024**3
    )

    plan = PlacementPlan(
        model_info=model_info,
        shared_device="cuda:0",
        hot_expert_slots=99,
        warm_expert_slots=138,
        cold_expert_count=19,
        gpu_utilization_pct=85.0,
        cpu_utilization_pct=60.0,
        estimated_load_time_sec=75.0
    )

    assert plan.model_info.is_moe == True
    assert plan.hot_expert_slots == 99
    assert plan.total_expert_count == 32 * 8  # layers * experts


def test_placement_plan_dense_model():
    from sparse_llm.loading.placement_plan import PlacementPlan
    from sparse_llm.loading.model_introspector import ModelInfo

    model_info = ModelInfo(
        model_id="llama-test",
        is_moe=False,
        num_layers=32,
        num_experts=0,
        num_experts_per_tok=0,
        shared_weight_bytes=70*1024**3,
        expert_weight_bytes=0,
        total_bytes=70*1024**3
    )

    plan = PlacementPlan(
        model_info=model_info,
        shared_device="cuda:0",
        hot_expert_slots=0,
        warm_expert_slots=0,
        cold_expert_count=0,
        gpu_utilization_pct=95.0,
        cpu_utilization_pct=0.0,
        estimated_load_time_sec=30.0
    )

    assert plan.model_info.is_moe == False
    assert plan.total_expert_count == 0

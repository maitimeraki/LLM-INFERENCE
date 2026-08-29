from sparse_llm import ExpertCache, ExpertKey


def test_public_cache_api_supports_layer_aware_experts():
    cache = ExpertCache(max_experts=2)
    value = object()

    cache.put((0, 1), value)

    assert ExpertKey == tuple[int, int]
    assert cache.get((0, 1)) is value
    assert cache.stats()["cached_experts"] == 1

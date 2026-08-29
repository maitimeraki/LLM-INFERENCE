from sparse_llm import ExpertCache, ExpertKey


def test_package_exports_lightweight_cache_api():
    cache = ExpertCache(max_experts=1)

    cache.put((0, 0), "cpu-weights")

    assert ExpertKey == tuple[int, int]
    assert cache.contains((0, 0))
    assert cache.get((0, 0)) == "cpu-weights"

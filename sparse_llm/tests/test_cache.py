import pytest

from sparse_llm.cache import ExpertCache


def test_layer_aware_keys_do_not_collide():
    cache = ExpertCache(max_experts=2)
    cache.put((0, 3), "layer-zero")
    cache.put((1, 3), "layer-one")

    assert cache.get((0, 3)) == "layer-zero"
    assert cache.get((1, 3)) == "layer-one"
    assert cache.stats()["hits"] == 2


def test_integer_lookup_rejects_ambiguous_layer_match():
    cache = ExpertCache(max_experts=3)
    cache.put((0, 2), "a")
    cache.put((1, 2), "b")

    with pytest.raises(ValueError, match="ambiguous"):
        cache.get(2)


def test_lru_eviction_and_miss_counters():
    cache = ExpertCache(max_experts=2)
    cache.put((0, 0), "a")
    cache.put((0, 1), "b")
    assert cache.get((0, 0)) == "a"
    cache.put((0, 2), "c")

    assert not cache.contains((0, 1))
    assert cache.get((0, 1)) is None
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hit_rate"] == pytest.approx(1 / 2)


def test_integer_compatibility_resolves_one_layer_aware_entry():
    cache = ExpertCache(max_experts=2)
    cache.put((4, 7), "weights")

    assert cache.get(7) == "weights"
    assert cache.contains(7)


def test_direct_integer_entry_takes_precedence():
    cache = ExpertCache(max_experts=3)
    cache.put((4, 7), "layer-aware")
    cache.put(7, "legacy")

    assert cache.get(7) == "legacy"


def test_tuple_layer_id_conflict_is_rejected():
    cache = ExpertCache(max_experts=1)

    with pytest.raises(ValueError, match="layer_id"):
        cache.put((2, 3), "weights", layer_id=4)


def test_contains_does_not_change_access_statistics():
    cache = ExpertCache(max_experts=1)
    cache.put((0, 0), "weights")
    before = cache.stats()

    assert cache.contains((0, 0))
    assert cache.stats() == before


def test_clear_removes_entries_and_resets_statistics():
    cache = ExpertCache(max_experts=1)
    cache.put((0, 0), "weights")
    assert cache.get((0, 0)) == "weights"

    cache.clear()

    assert cache.get((0, 0)) is None
    assert cache.stats()["cached_experts"] == 0
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 1


def test_malformed_tuple_is_rejected():
    cache = ExpertCache(max_experts=1)

    with pytest.raises(ValueError, match="ExpertKey"):
        cache.get((1, 2, 3))

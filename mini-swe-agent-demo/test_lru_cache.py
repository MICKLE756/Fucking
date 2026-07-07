import pytest

from lru_cache import LRUCache


def test_capacity_eviction():
    cache = LRUCache(2)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    assert cache.get("a") == -1
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_put_update_updates_value_and_recency():
    cache = LRUCache(2)

    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    cache.put("c", 3)

    assert cache.get("a") == 10
    assert cache.get("b") == -1
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_get_recency_prevents_accessed_key_from_eviction():
    cache = LRUCache(2)

    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1

    cache.put("c", 3)

    assert cache.get("b") == -1
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_missing_key_returns_minus_one():
    cache = LRUCache(1)

    assert cache.get("missing") == -1


def test_zero_capacity_stores_nothing():
    cache = LRUCache(0)

    cache.put("a", 1)

    assert cache.get("a") == -1
    assert len(cache) == 0


def test_negative_capacity_raises_value_error():
    with pytest.raises(ValueError):
        LRUCache(-1)

from cache.ttl_cache import TTLCache


def test_ttl_cache_expires_values() -> None:
    now = [100.0]
    cache = TTLCache[str](ttl_seconds=10, time_func=lambda: now[0])
    cache.set("key", "value")

    assert cache.get("key") == "value"
    now[0] = 110.0
    assert cache.get("key") is None


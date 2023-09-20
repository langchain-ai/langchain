"""Light weight unit test that attempts to import UpstashRedisStore.
"""

def test_import_storage() -> None:
    from langchain.storage.upstash_redis import UpstashRedisStore

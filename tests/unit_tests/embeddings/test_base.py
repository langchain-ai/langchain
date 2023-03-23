from redis import Redis
from typing import Any, Dict, List, Optional

from langchain.cache import InMemoryEmbeddingsCache, RedisEmbeddingsCache
from langchain.embeddings.fake import FakeEmbeddings

def test_caching() -> None:
    """Test InMemoryEmbeddingsCache."""
    cache = InMemoryEmbeddingsCache()
    embeddings = FakeEmbeddings(size=10, embeddings_cache=cache)
    res = embeddings.embed_query("foo")
    assert cache.lookup("foo") is res
    test_data = [0.0] * 10
    cache.update('f00', test_data)
    res = embeddings.embed_query("f00")
    assert res == test_data

def test_redis_caching() -> None:
    """Test RedisEmbeddingsCache."""
    # mock redis
    class MockRedis(Redis):
        def __init__(self) -> None:
            self.cache = {} # type: ignore
    
        def ft(self) -> Any: # type: ignore
            return self

        def info(self) -> None: # type: ignore
            return None
        
        def create_index(self) -> None:
            return None
        
        def hset(self, key, mapping) -> None: # type: ignore
            self.cache[key] = mapping
        
        def hget(self, key, field) -> Any: # type: ignore
            try:
                return self.cache[key][field]
            except KeyError:
                return None
    
    _redis = MockRedis()
    cache = RedisEmbeddingsCache(_redis)
    embeddings = FakeEmbeddings(size=10, precision="float32", embeddings_cache=cache)
    res = embeddings.embed_query("foo")
    print(type(res[0]))
    assert cache.lookup("foo") == res
    test_data = [0.0] * 10
    cache.update('f00', test_data)
    res = embeddings.embed_query("f00")
    assert res == test_data
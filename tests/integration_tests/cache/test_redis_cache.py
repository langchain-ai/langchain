import pytest

import langchain
from langchain.cache import RedisCache
from tests.integration_tests.cache.test_gptcache import basic_llm_caching_behavior

try:
    from redis import Redis

    redis_installed = True
except ImportError:
    redis_installed = False


@pytest.mark.skipif(not redis_installed, reason="redis not installed")
def test_redis_caching() -> None:
    """Test Redis LLM caching."""
    langchain.llm_cache = langchain.llm_cache = RedisCache(redis_=Redis())
    basic_llm_caching_behavior()

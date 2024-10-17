"""
Test Memcached llm cache functionality. Requires running instance of Memcached on
localhost default port (11211) and pymemcache
"""

import pytest
from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation, LLMResult

from langchain_community.cache import MemcachedCache
from tests.unit_tests.llms.fake_llm import FakeLLM

DEFAULT_MEMCACHED_URL = "localhost"


@pytest.mark.requires("pymemcache")
def test_memcached_cache() -> None:
    """Test general Memcached caching"""
    from pymemcache import Client

    set_llm_cache(MemcachedCache(Client(DEFAULT_MEMCACHED_URL)))
    llm = FakeLLM()

    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo"])
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    get_llm_cache().clear()


@pytest.mark.requires("pymemcache")
def test_memcached_cache_flush() -> None:
    """Test flushing Memcached cache"""
    from pymemcache import Client

    set_llm_cache(MemcachedCache(Client(DEFAULT_MEMCACHED_URL)))
    llm = FakeLLM()

    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo"])
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    get_llm_cache().clear(delay=0, noreply=False)

    # After cache has been cleared, the result shouldn't be the same
    output = llm.generate(["foo"])
    assert output != expected_output

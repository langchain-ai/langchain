import os
from typing import Any, Callable, Optional

import pytest

import langchain
from langchain.cache import GPTCache
from langchain.schema import Generation
from tests.unit_tests.llms.fake_llm import FakeLLM

try:
    from gptcache import Cache  # noqa: F401
    from gptcache.manager.factory import get_data_manager
    from gptcache.processor.pre import get_prompt

    gptcache_installed = True
except ImportError:
    gptcache_installed = False


def init_gptcache_map(cache_obj: Cache) -> None:
    i = getattr(init_gptcache_map, "_i", 0)
    cache_path = f"data_map_{i}.txt"
    if os.path.isfile(cache_path):
        os.remove(cache_path)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=get_data_manager(data_path=cache_path),
    )
    init_gptcache_map._i = i + 1  # type: ignore


@pytest.mark.skipif(not gptcache_installed, reason="gptcache not installed")
@pytest.mark.parametrize("init_func", [None, init_gptcache_map])
def test_gptcache_caching(init_func: Optional[Callable[[Any], None]]) -> None:
    """Test gptcache default caching behavior."""
    langchain.llm_cache = GPTCache(init_func)
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    langchain.llm_cache.update("foo", llm_string, [Generation(text="fizz")])
    _ = llm.generate(["foo", "bar", "foo"])
    cache_output = langchain.llm_cache.lookup("foo", llm_string)
    assert cache_output == [Generation(text="fizz")]

    langchain.llm_cache.clear()
    assert langchain.llm_cache.lookup("bar", llm_string) is None

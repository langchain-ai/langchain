import os

import pytest

import langchain
from langchain.cache import GPTCache
from langchain.schema import Generation, LLMResult
from tests.unit_tests.llms.fake_llm import FakeLLM

try:
    import gptcache  # noqa: F401

    gptcache_installed = True
except ImportError:
    gptcache_installed = False


@pytest.mark.skipif(not gptcache_installed, reason="gptcache not installed")
def test_gptcache_map_caching() -> None:
    """Test gptcache caching behavior."""

    from gptcache import Cache
    from gptcache.manager.factory import get_data_manager
    from gptcache.processor.pre import get_prompt

    i = 0
    file_prefix = "data_map"

    def init_gptcache_map(cache_obj: Cache) -> None:
        nonlocal i
        cache_path = f"{file_prefix}_{i}.txt"
        if os.path.isfile(cache_path):
            os.remove(cache_path)
        cache_obj.init(
            pre_embedding_func=get_prompt,
            data_manager=get_data_manager(data_path=cache_path),
        )
        i += 1

    langchain.llm_cache = GPTCache(init_gptcache_map)

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    langchain.llm_cache.update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo", "bar", "foo"])
    expected_cache_output = [Generation(text="foo")]
    cache_output = langchain.llm_cache.lookup("bar", llm_string)
    assert cache_output == expected_cache_output
    langchain.llm_cache = None
    expected_generations = [
        [Generation(text="fizz")],
        [Generation(text="foo")],
        [Generation(text="fizz")],
    ]
    expected_output = LLMResult(
        generations=expected_generations,
        llm_output=None,
    )
    assert output == expected_output

import os
from typing import Any, Callable, Union

import pytest
from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from langchain_community.cache import GPTCache
from tests.unit_tests.llms.fake_llm import FakeLLM

try:
    from gptcache import Cache  # noqa: F401
    from gptcache.manager.factory import get_data_manager
    from gptcache.processor.pre import get_prompt

    gptcache_installed = True
except ImportError:
    gptcache_installed = False


def init_gptcache_map(cache_obj: Any) -> None:
    i = getattr(init_gptcache_map, "_i", 0)
    cache_path = f"data_map_{i}.txt"
    if os.path.isfile(cache_path):
        os.remove(cache_path)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=get_data_manager(data_path=cache_path),
    )
    init_gptcache_map._i = i + 1  # type: ignore[attr-defined]


def init_gptcache_map_with_llm(cache_obj: Any, llm: str) -> None:
    cache_path = f"data_map_{llm}.txt"
    if os.path.isfile(cache_path):
        os.remove(cache_path)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=get_data_manager(data_path=cache_path),
    )


@pytest.mark.skipif(not gptcache_installed, reason="gptcache not installed")
@pytest.mark.parametrize(
    "init_func", [None, init_gptcache_map, init_gptcache_map_with_llm]
)
def test_gptcache_caching(
    init_func: Union[Callable[[Any, str], None], Callable[[Any], None], None],
) -> None:
    """Test gptcache default caching behavior."""
    set_llm_cache(GPTCache(init_func))
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    _ = llm.generate(["foo", "bar", "foo"])
    cache_output = get_llm_cache().lookup("foo", llm_string)
    assert cache_output == [Generation(text="fizz")]

    get_llm_cache().clear()
    assert get_llm_cache().lookup("bar", llm_string) is None

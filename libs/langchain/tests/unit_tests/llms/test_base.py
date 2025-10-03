"""Test base LLM functionality."""

from langchain_core.caches import InMemoryCache
from langchain_core.outputs import Generation, LLMResult

from langchain.globals import get_llm_cache, set_llm_cache
from langchain.llms.base import __all__
from tests.unit_tests.llms.fake_llm import FakeLLM

EXPECTED_ALL = [
    "BaseLLM",
    "LLM",
    "BaseLanguageModel",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


def test_caching() -> None:
    """Test caching behavior."""
    set_llm_cache(InMemoryCache())
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    cache = get_llm_cache()
    assert cache is not None
    cache.update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo", "bar", "foo"])
    expected_cache_output = [Generation(text="foo")]
    cache_output = cache.lookup("bar", llm_string)
    assert cache_output == expected_cache_output
    set_llm_cache(None)
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

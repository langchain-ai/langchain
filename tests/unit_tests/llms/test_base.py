"""Test base LLM functionality."""
import langchain
from langchain.cache import InMemoryCache
from langchain.schema import Generation, LLMResult
from tests.unit_tests.llms.fake_llm import FakeLLM


def test_caching() -> None:
    """Test caching behavior."""
    langchain.llm_cache = InMemoryCache()
    llm = FakeLLM()
    params = llm._llm_dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    langchain.llm_cache.update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo", "bar", "foo"])
    langchain.llm_cache = None
    expected_generations = [
        [Generation(text="fizz")],
        [Generation(text="foo")],
        [Generation(text="fizz")],
    ]
    expected_output = LLMResult(
        expected_generations,
        llm_output=None,
    )
    assert output == expected_output

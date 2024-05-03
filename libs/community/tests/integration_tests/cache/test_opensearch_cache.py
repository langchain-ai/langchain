from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from langchain_community.cache import OpenSearchSemanticCache
from tests.integration_tests.cache.fake_embeddings import (
    FakeEmbeddings,
)
from tests.unit_tests.llms.fake_llm import FakeLLM

DEFAULT_OPENSEARCH_URL = "http://localhost:9200"


def test_opensearch_semantic_cache() -> None:
    """Test opensearch semantic cache functionality."""
    set_llm_cache(
        OpenSearchSemanticCache(
            embedding=FakeEmbeddings(),
            opensearch_url=DEFAULT_OPENSEARCH_URL,
            score_threshold=0.0,
        )
    )
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    get_llm_cache().clear(llm_string=llm_string)
    output = get_llm_cache().lookup("bar", llm_string)
    assert output != [Generation(text="fizz")]


def test_opensearch_semantic_cache_multi() -> None:
    set_llm_cache(
        OpenSearchSemanticCache(
            embedding=FakeEmbeddings(),
            opensearch_url=DEFAULT_OPENSEARCH_URL,
            score_threshold=0.0,
        )
    )

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)
    output = get_llm_cache().lookup("bar", llm_string)
    assert output != [Generation(text="fizz"), Generation(text="Buzz")]

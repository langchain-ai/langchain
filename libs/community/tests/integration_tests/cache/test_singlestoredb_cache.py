"""Test SingleStoreDB semantic cache. Requires a SingleStore DB database.

Required to run this test:
    - a recent `singlestoredb` Python package available
    - a SingleStore DB instance;
"""

from importlib.util import find_spec

import pytest
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from langchain_community.cache import SingleStoreDBSemanticCache
from tests.integration_tests.cache.fake_embeddings import FakeEmbeddings
from tests.unit_tests.llms.fake_llm import FakeLLM

TEST_SINGLESTOREDB_URL = "root:pass@localhost:3306/db"

singlestoredb_installed = find_spec("singlestoredb") is not None


@pytest.mark.skipif(not singlestoredb_installed, reason="singlestoredb not installed")
def test_tinglestoredb_semantic_cache() -> None:
    """Test opensearch semantic cache functionality."""
    set_llm_cache(
        SingleStoreDBSemanticCache(
            embedding=FakeEmbeddings(),
            host=TEST_SINGLESTOREDB_URL,
            search_threshold=0.0,
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

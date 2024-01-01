"""
Test AstraDB caches. Requires an Astra DB vector instance.

Required to run this test:
    - a recent `astrapy` Python package available
    - an Astra DB instance;
    - the two environment variables set:
        export ASTRA_DB_API_ENDPOINT="https://<DB-ID>-us-east1.apps.astra.datastax.com"
        export ASTRA_DB_APPLICATION_TOKEN="AstraCS:........."
    - optionally this as well (otherwise defaults are used):
        export ASTRA_DB_KEYSPACE="my_keyspace"
"""
import os
from typing import Iterator

import pytest
from langchain_core.outputs import Generation, LLMResult

from langchain.cache import AstraDBCache, AstraDBSemanticCache
from langchain.globals import get_llm_cache, set_llm_cache
from tests.integration_tests.cache.fake_embeddings import FakeEmbeddings
from tests.unit_tests.llms.fake_llm import FakeLLM


def _has_env_vars() -> bool:
    return all(
        [
            "ASTRA_DB_APPLICATION_TOKEN" in os.environ,
            "ASTRA_DB_API_ENDPOINT" in os.environ,
        ]
    )


@pytest.fixture(scope="module")
def astradb_cache() -> Iterator[AstraDBCache]:
    cache = AstraDBCache(
        collection_name="lc_integration_test_cache",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    yield cache
    cache.astra_db.delete_collection("lc_integration_test_cache")


@pytest.fixture(scope="module")
def astradb_semantic_cache() -> Iterator[AstraDBSemanticCache]:
    fake_embe = FakeEmbeddings()
    sem_cache = AstraDBSemanticCache(
        collection_name="lc_integration_test_sem_cache",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        embedding=fake_embe,
    )
    yield sem_cache
    sem_cache.astra_db.delete_collection("lc_integration_test_cache")


@pytest.mark.requires("astrapy")
@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBCaches:
    def test_astradb_cache(self, astradb_cache: AstraDBCache) -> None:
        set_llm_cache(astradb_cache)
        llm = FakeLLM()
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
        output = llm.generate(["foo"])
        print(output)
        expected_output = LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        print(expected_output)
        assert output == expected_output
        astradb_cache.clear()

    def test_cassandra_semantic_cache(
        self, astradb_semantic_cache: AstraDBSemanticCache
    ) -> None:
        set_llm_cache(astradb_semantic_cache)
        llm = FakeLLM()
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
        output = llm.generate(["bar"])  # same embedding as 'foo'
        expected_output = LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        assert output == expected_output
        # clear the cache
        astradb_semantic_cache.clear()
        output = llm.generate(["bar"])  # 'fizz' is erased away now
        assert output != expected_output
        astradb_semantic_cache.clear()

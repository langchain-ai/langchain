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
from typing import AsyncIterator, Iterator

import pytest
from langchain_community.utilities.astradb import SetupMode
from langchain_core.caches import BaseCache
from langchain_core.language_models import LLM
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
    cache.collection.astra_db.delete_collection("lc_integration_test_cache")


@pytest.fixture
async def async_astradb_cache() -> AsyncIterator[AstraDBCache]:
    cache = AstraDBCache(
        collection_name="lc_integration_test_cache_async",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        setup_mode=SetupMode.ASYNC,
    )
    yield cache
    await cache.async_collection.astra_db.delete_collection(
        "lc_integration_test_cache_async"
    )


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
    sem_cache.collection.astra_db.delete_collection("lc_integration_test_sem_cache")


@pytest.fixture
async def async_astradb_semantic_cache() -> AsyncIterator[AstraDBSemanticCache]:
    fake_embe = FakeEmbeddings()
    sem_cache = AstraDBSemanticCache(
        collection_name="lc_integration_test_sem_cache_async",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        embedding=fake_embe,
        setup_mode=SetupMode.ASYNC,
    )
    yield sem_cache
    sem_cache.collection.astra_db.delete_collection(
        "lc_integration_test_sem_cache_async"
    )


@pytest.mark.requires("astrapy")
@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDBCaches:
    def test_astradb_cache(self, astradb_cache: AstraDBCache) -> None:
        self.do_cache_test(FakeLLM(), astradb_cache, "foo")

    async def test_astradb_cache_async(self, async_astradb_cache: AstraDBCache) -> None:
        await self.ado_cache_test(FakeLLM(), async_astradb_cache, "foo")

    def test_astradb_semantic_cache(
        self, astradb_semantic_cache: AstraDBSemanticCache
    ) -> None:
        llm = FakeLLM()
        self.do_cache_test(llm, astradb_semantic_cache, "bar")
        output = llm.generate(["bar"])  # 'fizz' is erased away now
        assert output != LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        astradb_semantic_cache.clear()

    async def test_astradb_semantic_cache_async(
        self, async_astradb_semantic_cache: AstraDBSemanticCache
    ) -> None:
        llm = FakeLLM()
        await self.ado_cache_test(llm, async_astradb_semantic_cache, "bar")
        output = await llm.agenerate(["bar"])  # 'fizz' is erased away now
        assert output != LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        await async_astradb_semantic_cache.aclear()

    @staticmethod
    def do_cache_test(llm: LLM, cache: BaseCache, prompt: str) -> None:
        set_llm_cache(cache)
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
        output = llm.generate([prompt])
        expected_output = LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        assert output == expected_output
        # clear the cache
        cache.clear()

    @staticmethod
    async def ado_cache_test(llm: LLM, cache: BaseCache, prompt: str) -> None:
        set_llm_cache(cache)
        params = llm.dict()
        params["stop"] = None
        llm_string = str(sorted([(k, v) for k, v in params.items()]))
        await get_llm_cache().aupdate("foo", llm_string, [Generation(text="fizz")])
        output = await llm.agenerate([prompt])
        expected_output = LLMResult(
            generations=[[Generation(text="fizz")]],
            llm_output={},
        )
        assert output == expected_output
        # clear the cache
        await cache.aclear()

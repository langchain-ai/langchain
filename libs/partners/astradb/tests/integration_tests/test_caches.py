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
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, cast

import pytest
from langchain_core.caches import BaseCache
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.pydantic_v1 import validator

from langchain_astradb import AstraDBCache, AstraDBSemanticCache
from langchain_astradb.utils.astradb import SetupMode


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    queries: Optional[Mapping] = None
    sequential_responses: Optional[bool] = False
    response_index: int = 0

    @validator("queries", always=True)
    def check_queries_required(
        cls, queries: Optional[Mapping], values: Mapping[str, Any]
    ) -> Optional[Mapping]:
        if values.get("sequential_response") and not queries:
            raise ValueError(
                "queries is required when sequential_response is set to True"
            )
        return queries

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens."""
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.sequential_responses:
            return self._get_next_response_in_sequence
        if self.queries is not None:
            return self.queries[prompt]
        if stop is None:
            return "foo"
        else:
            return "bar"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _get_next_response_in_sequence(self) -> str:
        queries = cast(Mapping, self.queries)
        response = queries[list(queries.keys())[self.response_index]]
        self.response_index = self.response_index + 1
        return response


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

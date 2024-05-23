"""Test Azure CosmosDB cache functionality.

Required to run this test:
    - a recent 'pymongo' Python package available
    - an Azure CosmosDB Mongo vCore instance
    - one environment variable set:
        export MONGODB_VCORE_URI="connection string for azure cosmos db mongo vCore"
"""

import os
import uuid

import pytest
from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from langchain_community.cache import AzureCosmosDBSemanticCache
from langchain_community.vectorstores.azure_cosmos_db import (
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from tests.integration_tests.cache.fake_embeddings import (
    FakeEmbeddings,
)
from tests.unit_tests.llms.fake_llm import FakeLLM

INDEX_NAME = "langchain-test-index"
NAMESPACE = "langchain_test_db.langchain_test_collection"
CONNECTION_STRING: str = os.environ.get("MONGODB_VCORE_URI", "")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

num_lists = 3
dimensions = 10
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_IVF
m = 16
ef_construction = 64
ef_search = 40
score_threshold = 0.1
application_name = "LANGCHAIN_CACHING_PYTHON"


def _has_env_vars() -> bool:
    return all(["MONGODB_VCORE_URI" in os.environ])


def random_string() -> str:
    return str(uuid.uuid4())


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache() -> None:
    set_llm_cache(
        AzureCosmosDBSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            cosmosdb_client=None,
            embedding=FakeEmbeddings(),
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=kind,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_inner_product() -> None:
    set_llm_cache(
        AzureCosmosDBSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            cosmosdb_client=None,
            embedding=FakeEmbeddings(),
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=kind,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi() -> None:
    set_llm_cache(
        AzureCosmosDBSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            cosmosdb_client=None,
            embedding=FakeEmbeddings(),
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=kind,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
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


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi_inner_product() -> None:
    set_llm_cache(
        AzureCosmosDBSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            cosmosdb_client=None,
            embedding=FakeEmbeddings(),
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=kind,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
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


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_hnsw() -> None:
    set_llm_cache(
        AzureCosmosDBSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            cosmosdb_client=None,
            embedding=FakeEmbeddings(),
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_inner_product_hnsw() -> None:
    set_llm_cache(
        AzureCosmosDBSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            cosmosdb_client=None,
            embedding=FakeEmbeddings(),
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi_hnsw() -> None:
    set_llm_cache(
        AzureCosmosDBSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            cosmosdb_client=None,
            embedding=FakeEmbeddings(),
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
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


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi_inner_product_hnsw() -> None:
    set_llm_cache(
        AzureCosmosDBSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            cosmosdb_client=None,
            embedding=FakeEmbeddings(),
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
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

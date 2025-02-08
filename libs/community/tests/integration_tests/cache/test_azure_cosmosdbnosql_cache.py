"""Test` Azure CosmosDB NoSql cache functionality."""

from typing import Any

import pytest
from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from langchain_community.cache import AzureCosmosDBNoSqlSemanticCache
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings
from tests.unit_tests.llms.fake_llm import FakeLLM

HOST = "COSMOS_DB_URI"
KEY = "COSMOS_DB_KEY"


@pytest.fixture()
def cosmos_client() -> Any:
    from azure.cosmos import CosmosClient

    return CosmosClient(HOST, KEY)


@pytest.fixture()
def partition_key() -> Any:
    from azure.cosmos import PartitionKey

    return PartitionKey(path="/id")


# cosine, euclidean, innerproduct
def indexing_policy(index_type: str) -> dict:
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": index_type}],
    }


def vector_embedding_policy(distance_function: str) -> dict:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": distance_function,
                "dimensions": 1536,
            }
        ]
    }


def test_azure_cosmos_db_nosql_semantic_cache_cosine_quantizedflat(
    cosmos_client: Any,
    partition_key: Any,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy("cosine"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
            full_text_search_fields=["text"],
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


def test_azure_cosmos_db_nosql_semantic_cache_cosine_disk_ann(
    cosmos_client: Any,
    partition_key: Any,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy("cosine"),
            indexing_policy=indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
            full_text_search_fields=["text"],
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


def test_azure_cosmos_db_nosql_semantic_cache_dotproduct_quantizedflat(
    cosmos_client: Any,
    partition_key: Any,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy("dotproduct"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
            full_text_search_fields=["text"],
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


def test_azure_cosmos_db_nosql_semantic_cache_dotproduct_disk_ann(
    cosmos_client: Any,
    partition_key: Any,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy("dotproduct"),
            indexing_policy=indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
            full_text_search_fields=["text"],
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


def test_azure_cosmos_db_nosql_semantic_cache_euclidean_quantizedflat(
    cosmos_client: Any,
    partition_key: Any,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy("euclidean"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
            full_text_search_fields=["text"],
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


def test_azure_cosmos_db_nosql_semantic_cache_euclidean_disk_ann(
    cosmos_client: Any,
    partition_key: Any,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy("euclidean"),
            indexing_policy=indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
            full_text_search_fields=["text"],
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

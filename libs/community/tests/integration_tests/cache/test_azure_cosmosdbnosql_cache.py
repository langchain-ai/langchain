"""Test Azure CosmosDB NoSql cache functionality.
"""
import os
import uuid

import pytest
from azure.cosmos import CosmosClient, PartitionKey
from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from langchain_community.cache import AzureCosmosDBNoSqlSemanticCache
from langchain_community.vectorstores import AzureCosmosDBNoSqlVectorSearch

from libs.community.tests.integration_tests.cache.fake_embeddings import (
    FakeEmbeddings,
)
from libs.community.tests.unit_tests.llms.fake_llm import FakeLLM


URI = 'COSMOSDB_URI'
KEY = 'COSMOSDB_KEY'
test_client = CosmosClient(URL, credential=KEY)

indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
}

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 1536,
        }
    ]
}

partition_key = PartitionKey(path="/id")
cosmos_container_properties_test = {"partition_key": partition_key}
cosmos_database_properties_test = {}

# @pytest.fixture(scope="session")
def test_azure_cosmos_db_nosql_semantic_cache() -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=test_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
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


def test_azure_cosmos_db_semantic_cache_inner_product() -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=test_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
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


def test_azure_cosmos_db_semantic_cache_multi() -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=test_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
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



def test_azure_cosmos_db_semantic_cache_multi_inner_product() -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=test_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
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



def test_azure_cosmos_db_semantic_cache_hnsw() -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=test_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
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



def test_azure_cosmos_db_semantic_cache_inner_product_hnsw() -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=test_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
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



def test_azure_cosmos_db_semantic_cache_multi_hnsw() -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=test_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
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



def test_azure_cosmos_db_semantic_cache_multi_inner_product_hnsw() -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=test_client,
            embedding=FakeEmbeddings(),
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
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

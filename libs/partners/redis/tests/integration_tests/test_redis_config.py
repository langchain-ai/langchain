import uuid
from typing import Any, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from redis import Redis
from redisvl.query import CountQuery  # type: ignore
from redisvl.query.filter import FilterExpression  # type: ignore
from redisvl.schema import IndexSchema, StorageType  # type: ignore

from langchain_redis import RedisConfig, RedisVectorStore


@pytest.fixture
def redis_url() -> str:
    return "redis://localhost:6379"


@pytest.fixture
def embeddings() -> Embeddings:
    return OpenAIEmbeddings()


@pytest.fixture
def sample_texts() -> List[str]:
    return ["foo", "bar", "baz"]


def test_redis_config_default_initialization() -> None:
    config = RedisConfig()
    assert config.key_prefix == config.index_name
    assert config.redis_url == "redis://localhost:6379"
    assert config.distance_metric == "COSINE"
    assert config.indexing_algorithm == "FLAT"
    assert config.vector_datatype == "FLOAT32"
    assert config.storage_type == "hash"
    assert config.content_field == "text"
    assert config.embedding_field == "embedding"
    assert config.index_schema is None


def test_redis_config_custom_initialization() -> None:
    config = RedisConfig(
        index_name="custom_index",
        key_prefix="custom_prefix",
        redis_url="redis://custom:6379",
        distance_metric="L2",
        indexing_algorithm="HNSW",
        vector_datatype="FLOAT64",
        storage_type="json",
        content_field="content",
        embedding_field="vector",
    )
    assert config.index_name == "custom_index"
    assert config.key_prefix == "custom_prefix"
    assert config.redis_url == "redis://custom:6379"
    assert config.distance_metric == "L2"
    assert config.indexing_algorithm == "HNSW"
    assert config.vector_datatype == "FLOAT64"
    assert config.storage_type == "json"
    assert config.content_field == "content"
    assert config.embedding_field == "vector"


def test_redis_config_with_metadata_schema() -> None:
    metadata_schema = [
        {"name": "color", "type": "tag"},
        {"name": "price", "type": "numeric"},
    ]
    config = RedisConfig(metadata_schema=metadata_schema)
    assert config.metadata_schema == metadata_schema


def test_redis_config_to_index_schema(embeddings: Embeddings) -> None:
    config = RedisConfig(index_name="test_index")
    config.embedding_dimensions = len(embeddings.embed_query("test"))
    schema = config.to_index_schema()
    assert isinstance(schema, IndexSchema)
    assert schema.index.name == "test_index"
    assert schema.index.storage_type == StorageType.HASH
    assert len(schema.fields) == 3  # id_field, content, embedding
    field_names = [field.name for field in schema.fields.values()]
    assert config.id_field in field_names
    assert config.content_field in field_names
    assert config.embedding_field in field_names


def test_redis_config_from_existing_index(redis_url: str) -> None:
    # First, create an index
    index_name = f"test_index_{uuid.uuid4().hex}"
    vector_store = RedisVectorStore.from_texts(
        ["test"], OpenAIEmbeddings(), index_name=index_name, redis_url=redis_url
    )

    # Now, create a config from the existing index
    redis_client = Redis.from_url(redis_url)
    config = RedisConfig.from_existing_index(index_name, redis_client)

    assert config.index_name == index_name
    assert config.storage_type == "hash"
    assert config.embedding_field == "embedding"

    # Clean up
    vector_store.index.delete(drop=True)


def test_redis_vector_store_with_config(
    redis_url: str, embeddings: Embeddings, sample_texts: List[str]
) -> None:
    config = RedisConfig(index_name=f"test_index_{uuid.uuid4().hex}")
    vector_store = RedisVectorStore(embeddings, config=config)
    vector_store.add_texts(sample_texts)

    count_query = CountQuery(FilterExpression("*"))
    count = vector_store.index.query(count_query)
    assert count == len(sample_texts)

    # Clean up
    vector_store.index.delete(drop=True)


def test_redis_vector_store_from_texts_with_config(
    redis_url: str, embeddings: Embeddings, sample_texts: List[str]
) -> None:
    config = RedisConfig(index_name=f"test_index_{uuid.uuid4().hex}")
    vector_store = RedisVectorStore.from_texts(sample_texts, embeddings, config=config)

    count_query = CountQuery(FilterExpression("*"))
    count = vector_store.index.query(count_query)
    assert count == len(sample_texts)

    # Clean up
    vector_store.index.delete(drop=True)


def test_redis_vector_store_from_documents_with_config(
    redis_url: str, embeddings: Embeddings
) -> None:
    docs = [Document(page_content="test", metadata={"key": "value"})]
    metadata_schema = [
        {"name": "key", "type": "tag"},
    ]
    config = RedisConfig(metadata_schema=metadata_schema)
    vector_store = RedisVectorStore.from_documents(docs, embeddings, config=config)

    results = vector_store.similarity_search("test", k=1)
    assert len(results) == 1
    assert results[0].page_content == "test"
    assert results[0].metadata["key"] == "value"

    # Clean up
    vector_store.index.delete(drop=True)


def test_redis_config_with_custom_schema(
    redis_url: str, embeddings: Embeddings
) -> None:
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": f"test_index_{uuid.uuid4().hex}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "text", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "dims": 1536,
                        "distance_metric": "cosine",
                        "algorithm": "FLAT",
                    },
                },
                {"name": "category", "type": "tag"},
            ],
        }
    )

    config = RedisConfig.from_schema(schema)
    vector_store = RedisVectorStore(embeddings, config=config)

    vector_store.add_texts(["test"], [{"category": "test_category"}])
    results = vector_store.similarity_search("test", k=1, return_metadata=True)

    assert len(results) == 1
    assert results[0].page_content == "test"
    assert results[0].metadata["category"] == "test_category"

    # Clean up
    vector_store.index.delete(drop=True)


def test_redis_config_with_yaml_schema(
    tmp_path: Any, redis_url: str, embeddings: Embeddings
) -> None:
    schema_path = tmp_path / "test_schema.yaml"
    schema_content = """
    index:
      name: test_index
      storage_type: hash
    fields:
      - name: text
        type: text
      - name: embedding
        type: vector
        attrs:
          dims: 1536
          distance_metric: cosine
          algorithm: FLAT
      - name: category
        type: tag
    """
    schema_path.write_text(schema_content)

    config = RedisConfig.from_yaml(str(schema_path))
    vector_store = RedisVectorStore(embeddings, config=config)

    vector_store.add_texts(["test"], [{"category": "test_category"}])
    results = vector_store.similarity_search("test", k=1)
    assert len(results) == 1
    assert results[0].page_content == "test"
    assert results[0].metadata["category"] == "test_category"

    # Clean up
    vector_store.index.delete(drop=True)

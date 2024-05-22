from typing import Any, Dict, Iterable, List, Optional
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_redis import RedisConfig, RedisVectorStore


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


class MockField:
    def __init__(
        self, name: str, field_type: str, attrs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.name = name
        self.type = field_type
        self.attrs = attrs or {}


class MockSchema:
    def __init__(self, fields: Dict[str, Dict[str, Any]]) -> None:
        self.fields = {
            name: MockField(name, details["type"], details.get("attrs"))
            for name, details in fields.items()
        }

    def values(self) -> Iterable[MockField]:
        return self.fields.values()


class MockStorage:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}

    def get(self, client: Any, doc_ids: List[str]) -> List[Dict[str, Any]]:
        return [self.data.get(doc_id, {}) for doc_id in doc_ids]


class MockSearchIndex:
    def __init__(
        self, schema: Optional[Dict[str, Any]] = None, client: Optional[Any] = None
    ) -> None:
        self.data: List[Dict[str, Any]] = []
        default_schema = {
            "text": {"type": "text"},
            "embedding": {
                "type": "vector",
                "attrs": {"dims": 3, "distance_metric": "cosine"},
            },
            "metadata": {"type": "text"},
        }
        self.schema = MockSchema(
            schema["fields"] if schema and "fields" in schema else default_schema
        )
        self.client = client or Mock()
        self._storage = MockStorage()

    def create(self, overwrite: bool = False) -> None:
        pass

    def load(
        self, documents: List[Dict[str, Any]], keys: Optional[List[str]] = None
    ) -> List[str]:
        for i, doc in enumerate(documents):
            key = keys[i] if keys else f"key_{i}"
            self._storage.data[key] = doc
        self.data.extend(documents)
        return keys or [f"key_{i}" for i in range(len(documents))]

    def query(self, query: Any) -> List[Dict[str, Any]]:
        k = query._num_results if hasattr(query, "_num_results") else len(self.data)
        return [
            {
                "id": f"key_{i}",
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "vector_distance": 0.1,
            }
            for i, doc in enumerate(self.data[:k])
        ]

    def delete(self, keys: List[str]) -> None:
        self.data = [doc for i, doc in enumerate(self.data) if f"key_{i}" not in keys]
        for key in keys:
            self._storage.data.pop(key, None)

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> "MockSearchIndex":
        return cls(schema=dict_data)

    def set_client(self, client: Any) -> None:
        self.client = client

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MockSearchIndex":
        return cls()


class TestRedisVectorStore:
    @pytest.fixture
    def mock_embeddings(self) -> MockEmbeddings:
        return MockEmbeddings()

    @pytest.fixture
    def mock_config(self) -> RedisConfig:
        config = RedisConfig(
            index_name="test_index",
            redis_url="redis://localhost:6379",
            schema_path="mock_schema.yaml",
        )
        config.content_field = "text"
        config.embedding_field = "embedding"
        config.index_schema = {
            "fields": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "vector",
                    "attrs": {"dims": 3, "distance_metric": "cosine"},
                },
                "metadata": {"type": "text"},
            }
        }
        return config

    @pytest.fixture
    def vector_store(
        self, mock_embeddings: MockEmbeddings, mock_config: RedisConfig
    ) -> RedisVectorStore:
        with patch("langchain_redis.vectorstores.SearchIndex", MockSearchIndex):
            return RedisVectorStore(embeddings=mock_embeddings, config=mock_config)

    def test_add_texts(self, vector_store: RedisVectorStore) -> None:
        texts = ["Hello, world!", "Test document"]
        metadatas = [{"source": "greeting"}, {"source": "test"}]
        keys = vector_store.add_texts(texts, metadatas)
        assert len(keys) == 2
        assert all(key.startswith("key_") for key in keys)

    def test_similarity_search(self, vector_store: RedisVectorStore) -> None:
        vector_store.add_texts(["Hello, world!", "Test document"])
        results = vector_store.similarity_search("Hello", k=1)
        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].page_content in ["Hello, world!", "Test document"]

    def test_similarity_search_with_score(self, vector_store: RedisVectorStore) -> None:
        vector_store.add_texts(["Hello, world!", "Test document"])
        results = vector_store.similarity_search_with_score("Hello", k=1)
        assert len(results) == 1
        assert isinstance(results[0][0], Document)
        assert isinstance(results[0][1], float)

    def test_max_marginal_relevance_search(
        self, vector_store: RedisVectorStore
    ) -> None:
        vector_store.add_texts(["Hello, world!", "Test document", "Another test"])
        results = vector_store.max_marginal_relevance_search("Hello", k=2, fetch_k=3)
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_delete(self, vector_store: RedisVectorStore) -> None:
        keys = vector_store.add_texts(["Hello, world!", "Test document"])
        vector_store.delete(keys)
        results = vector_store.similarity_search("Hello", k=1)
        assert len(results) == 0

    @patch("langchain_redis.vectorstores.RedisVectorStore.add_texts")
    def test_from_texts(
        self,
        mock_add_texts: Mock,
        mock_embeddings: MockEmbeddings,
        mock_config: RedisConfig,
    ) -> None:
        with patch("langchain_redis.vectorstores.SearchIndex", MockSearchIndex):
            texts = ["Hello, world!", "Test document"]
            metadatas = [{"source": "greeting"}, {"source": "test"}]
            RedisVectorStore.from_texts(
                texts, mock_embeddings, metadatas=metadatas, config=mock_config
            )
            mock_add_texts.assert_called_once_with(texts, metadatas, None)

    @patch("langchain_redis.vectorstores.RedisVectorStore.add_texts")
    def test_from_documents(
        self,
        mock_add_texts: Mock,
        mock_embeddings: MockEmbeddings,
        mock_config: RedisConfig,
    ) -> None:
        with patch("langchain_redis.vectorstores.SearchIndex", MockSearchIndex):
            docs = [
                Document(page_content="Hello, world!", metadata={"source": "greeting"}),
                Document(page_content="Test document", metadata={"source": "test"}),
            ]
            RedisVectorStore.from_documents(docs, mock_embeddings, config=mock_config)
            mock_add_texts.assert_called_once_with(
                ["Hello, world!", "Test document"],
                [{"source": "greeting"}, {"source": "test"}],
                None,
            )

import time
from unittest.mock import Mock

import pytest
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.vectorstores import VectorStore

from langchain_pinecone import PineconeVectorStore
from langchain_pinecone._utilities import DistanceStrategy, FakeEncoder


@pytest.fixture
def index_name() -> str:
    return "hybrid-langchain-index"  # index must already exist


def test_initialization(
    index_name: str,
    fake_embedding_model: FakeEmbeddings,
    fake_encoder_model: FakeEncoder,
    distance_strategy: DistanceStrategy,
) -> None:
    assert PineconeVectorStore(index=index_name, embedding=fake_embedding_model)
    assert PineconeVectorStore(
        index=index_name,
        embedding=fake_embedding_model,
        sparse_encoder=fake_encoder_model,
        distance_strategy=distance_strategy,
    )


def test_from_text_initialization(
    index_name: str,
    fake_embedding_model: FakeEmbeddings,
    fake_encoder_model: FakeEncoder,
    distance_strategy: DistanceStrategy,
) -> None:
    texts = ["foobar"]
    assert PineconeVectorStore.from_texts(
        texts=texts,
        embedding=fake_embedding_model,
        index_name=index_name,
        distance_strategy=distance_strategy,
    )

    assert PineconeVectorStore.from_texts(
        texts=texts,
        embedding=fake_embedding_model,
        index_name=index_name,
        distance_strategy=distance_strategy,
        sparse_encoder=fake_encoder_model,
    )


def test_add_texts(
    hybrid_vector_store: PineconeVectorStore, semantic_vector_store: PineconeVectorStore
) -> None:
    texts = ["foo", "baz", "bar"]
    sleep = 5

    hybrid_output = hybrid_vector_store.add_texts(texts)
    semantic_output = semantic_vector_store.add_texts(texts)

    time.sleep(sleep)  # prevent race condition

    assert len(hybrid_output) == len(texts)
    assert len(semantic_output) == len(texts)


def test_similarity_search_by_vector_with_score(
    fake_embedding_model: Embeddings,
    fake_encoder_model: FakeEncoder,
    hybrid_vector_store: PineconeVectorStore,
    semantic_vector_store: PineconeVectorStore,
) -> None:
    k = 2
    alpha = 0.5
    query = "foobar"
    embedding = fake_embedding_model.embed_query(query)
    encoding = fake_encoder_model.encode_queries(query)
    assert (
        len(
            semantic_vector_store.similarity_search_by_vector_with_score(embedding, k=k)
        )
        == k
    )
    assert (
        len(
            hybrid_vector_store.similarity_search_by_vector_with_score(
                embedding, k=k, encoding=encoding, alpha=alpha
            )
        )
        == k
    )


def test_similarity_search_with_score(
    semantic_vector_store: PineconeVectorStore, hybrid_vector_store: PineconeVectorStore
) -> None:
    k = 3
    alpha = 0.3
    query = "foobar"
    assert (
        len(semantic_vector_store.similarity_search_with_score(query=query, k=k)) == k
    )
    assert (
        len(
            hybrid_vector_store.similarity_search_with_score(
                query=query, k=k, alpha=alpha
            )
        )
        == k
    )


def test_similarity_search(
    semantic_vector_store: PineconeVectorStore, hybrid_vector_store: PineconeVectorStore
) -> None:
    alpha = 0.5
    k = 3
    query = "foobar"
    assert len(semantic_vector_store.similarity_search(query=query, k=k)) == k
    assert (
        len(hybrid_vector_store.similarity_search(query=query, k=k, alpha=alpha)) == k
    )


def test_from_existing_index(
    index_name: str,
    fake_embedding_model: FakeEmbeddings,
    fake_encoder_model: FakeEncoder,
    distance_strategy: DistanceStrategy,
) -> None:
    assert isinstance(
        PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=fake_embedding_model,
            distance_strategy=distance_strategy,
        ),
        VectorStore,
    )
    assert isinstance(
        PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=fake_embedding_model,
            sparse_encoder=fake_encoder_model,
            distance_strategy=distance_strategy,
        ),
        VectorStore,
    )


@pytest.mark.parametrize(
    ("alpha, expected"),
    [
        pytest.param(-0.1, False, marks=pytest.mark.xfail),
        pytest.param(1.1, False, marks=pytest.mark.xfail),
    ],
)
def test_invalid_alphas(
    hybrid_vector_store: PineconeVectorStore, alpha: float, expected: bool
) -> None:
    hybrid_vector_store._alpha = alpha


@pytest.mark.parametrize(("alpha"), [(0, 0.5, 1)])
def test_valid_alphas(hybrid_vector_store: PineconeVectorStore, alpha: float) -> None:
    hybrid_vector_store._alpha = alpha


def test_id_prefix() -> None:
    """Test integration of the id_prefix parameter."""
    embedding = Mock()
    embedding.embed_documents = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    index = Mock()
    index.upsert = Mock(return_value=None)
    text_key = "testing"
    vectorstore = PineconeVectorStore(index, embedding, text_key)
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    id_prefix = "testing_prefixes"
    vectorstore.add_texts(texts, id_prefix=id_prefix, async_req=False)

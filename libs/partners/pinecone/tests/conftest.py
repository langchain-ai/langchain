import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings

from langchain_pinecone import PineconeVectorStore
from langchain_pinecone._utilities import DistanceStrategy, FakeEncoder


@pytest.fixture
def distance_strategy() -> DistanceStrategy:
    return DistanceStrategy.MAX_INNER_PRODUCT


@pytest.fixture
def fake_embedding_model() -> Embeddings:
    """Fake sparse encoder for testing"""
    size = 1536
    embedding = FakeEmbeddings(size=size)
    return embedding


@pytest.fixture
def fake_encoder_model() -> FakeEncoder:
    """Fake sparse encoder for testing"""
    seed = 2
    size = 10
    sparse_encoder = FakeEncoder(seed=seed, size=size)
    return sparse_encoder


@pytest.fixture
def semantic_vector_store(
    fake_embedding_model: Embeddings, index_name: str
) -> PineconeVectorStore:
    """Vector store for vector store"""
    return PineconeVectorStore(embedding=fake_embedding_model, index_name=index_name)


@pytest.fixture
def hybrid_vector_store(
    fake_embedding_model: Embeddings,
    fake_encoder_model: FakeEncoder,
    index_name: str,
    distance_strategy: DistanceStrategy,
) -> PineconeVectorStore:
    """Vector store for hybrid search"""
    return PineconeVectorStore(
        embedding=fake_embedding_model,
        sparse_encoder=fake_encoder_model,
        index_name=index_name,
        distance_strategy=distance_strategy,
    )

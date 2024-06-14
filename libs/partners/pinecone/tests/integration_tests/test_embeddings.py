"""Test embedding model integration."""

import pytest

from langchain_pinecone import PineconeEmbeddings

MODEL = "multilingual-e5-large"
MODEL_DIM = 1024

class TestPineconeEmbeddings:
    """Test Pinecone embeddings."""

    def test_embed_query(self) -> None:
        """Test embedding a query."""
        pinecone_embeddings = PineconeEmbeddings(model=MODEL)
        out = pinecone_embeddings.embed_query("Hello, world!")
        assert isinstance(out, list)
        assert len(out) == MODEL_DIM

    @pytest.mark.asyncio
    async def test_aembed_query(self) -> None:
        """Test embedding a query."""
        pinecone_embeddings = PineconeEmbeddings(model=MODEL)
        out = await pinecone_embeddings.aembed_query("Hello, world!")
        assert isinstance(out, list)
        assert len(out) == MODEL_DIM

    def test_embed_documents(self) -> None:
        """Test embedding documents."""
        pinecone_embeddings = PineconeEmbeddings(model=MODEL)
        out = pinecone_embeddings.embed_documents(["Hello, world!", "This is a test."])
        assert isinstance(out, list)
        assert len(out) == 2
        assert len(out[0]) == MODEL_DIM

    @pytest.mark.asyncio
    async def test_aembed_documents(self) -> None:
        """Test embedding documents."""
        pinecone_embeddings = PineconeEmbeddings(model=MODEL)
        out = await pinecone_embeddings.aembed_documents(["Hello, world!", "This is a test."])
        assert isinstance(out, list)
        assert len(out) == 2
        assert len(out[0]) == MODEL_DIM

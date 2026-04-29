"""Integration tests for Perplexity Embeddings API."""

import os

import pytest

from langchain_perplexity import PerplexityEmbeddings


@pytest.mark.skipif(not os.environ.get("PPLX_API_KEY"), reason="PPLX_API_KEY not set")
class TestPerplexityEmbeddings:
    def test_embed_documents(self) -> None:
        """Test embedding a list of documents."""
        embeddings = PerplexityEmbeddings()
        texts = ["hello world", "goodbye world"]
        vectors = embeddings.embed_documents(texts)

        assert len(vectors) == len(texts)
        assert all(isinstance(v, list) for v in vectors)
        assert all(len(v) > 0 for v in vectors)
        # All vectors should have the same dimensionality.
        assert len({len(v) for v in vectors}) == 1
        assert all(isinstance(x, float) for x in vectors[0])

    def test_embed_query(self) -> None:
        """Test embedding a single query."""
        embeddings = PerplexityEmbeddings()
        vector = embeddings.embed_query("What is the capital of France?")

        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(x, float) for x in vector)

    def test_embed_query_matches_documents_dim(self) -> None:
        """Embeddings from query and documents should share dimensionality."""
        embeddings = PerplexityEmbeddings()
        query_vec = embeddings.embed_query("hello")
        doc_vecs = embeddings.embed_documents(["hello"])
        assert len(query_vec) == len(doc_vecs[0])

    async def test_aembed_documents(self) -> None:
        """Test async embedding a list of documents."""
        embeddings = PerplexityEmbeddings()
        vectors = await embeddings.aembed_documents(["hello", "world"])
        assert len(vectors) == 2
        assert all(len(v) > 0 for v in vectors)

    async def test_aembed_query(self) -> None:
        """Test async embedding a single query."""
        embeddings = PerplexityEmbeddings()
        vector = await embeddings.aembed_query("hello")
        assert isinstance(vector, list)
        assert len(vector) > 0

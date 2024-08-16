"""Test AI21 embeddings."""

from langchain_ai21.embeddings import AI21Embeddings
from typing import Type
import pytest

from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests

class TestAI21Embeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[AI21Embeddings]:
        return AI21Embeddings

def test_langchain_ai21_embedding_documents__with_explicit_chunk_size() -> None:
    """Test AI21 embeddings with chunk size passed as an argument."""
    documents = ["foo", "bar"]
    embedding = AI21Embeddings()
    output = embedding.embed_documents(documents, batch_size=1)
    assert len(output) == 2
    assert len(output[0]) > 0

def test_langchain_ai21_embedding_query__with_explicit_chunk_size() -> None:
    """Test AI21 embeddings with chunk size passed as an argument."""
    documents = "foo bar"
    embedding = AI21Embeddings()
    output = embedding.embed_query(documents, batch_size=1)
    assert len(output) > 0

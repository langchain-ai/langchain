"""Test Nomic embeddings."""

from langchain_nomic.embeddings import NomicEmbeddings


from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests
from typing import Type

class TestNomicEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[NomicEmbeddings]:
        return NomicEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "llama3:latest"}


def test_langchain_nomic_embedding_dimensionality() -> None:
    """Test nomic embeddings."""
    documents = ["foo bar"]
    embedding = NomicEmbeddings(model="nomic-embed-text-v1.5", dimensionality=256)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 256

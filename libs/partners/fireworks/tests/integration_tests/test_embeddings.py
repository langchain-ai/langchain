"""Test Fireworks embeddings."""

from langchain_fireworks.embeddings import FireworksEmbeddings

from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests
from typing import Type

class TestFireworksEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[FireworksEmbeddings]:
        return FireworksEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nomic-ai/nomic-embed-text-v1.5"}

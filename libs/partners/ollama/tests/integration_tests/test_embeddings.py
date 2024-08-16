"""Test Ollama embeddings."""

from langchain_ollama.embeddings import OllamaEmbeddings

from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests
from typing import Type

class TestOllamaEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[OllamaEmbeddings]:
        return OllamaEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "llama3:latest"}

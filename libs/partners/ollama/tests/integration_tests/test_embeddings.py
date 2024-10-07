"""Test Ollama embeddings."""

from typing import Type

from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_ollama.embeddings import OllamaEmbeddings


class TestOllamaEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[OllamaEmbeddings]:
        return OllamaEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "llama3:latest"}

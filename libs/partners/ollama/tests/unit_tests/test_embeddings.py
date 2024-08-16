"""Test embedding model integration."""

from langchain_ollama.embeddings import OllamaEmbeddings


from langchain_standard_tests.unit_tests import EmbeddingsUnitTests
from typing import Type

class TestOllamaEmbeddings(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[OllamaEmbeddings]:
        return OllamaEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "llama3:latest"}

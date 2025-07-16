"""Test Ollama embeddings."""

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_ollama.embeddings import OllamaEmbeddings

MODEL_NAME = "llama3.1"


class TestOllamaEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[OllamaEmbeddings]:
        return OllamaEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": MODEL_NAME}

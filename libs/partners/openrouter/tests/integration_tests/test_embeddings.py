"""Test OpenRouter embeddings."""

import os

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_openrouter.embeddings import OpenRouterEmbeddings

MODEL_NAME = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")


class TestOpenRouterEmbeddings(EmbeddingsIntegrationTests):
    """Test class for OpenRouter embeddings integration tests."""

    @property
    def embeddings_class(self) -> type[OpenRouterEmbeddings]:
        """Return the embeddings class being tested."""
        return OpenRouterEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        """Return the model parameters for the embeddings class."""
        return {"model": MODEL_NAME}

from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from langchain_tests.integration_tests import EmbeddingsIntegrationTests
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestFakeEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return DeterministicFakeEmbedding

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return {"size": 6}  # embedding dimension


class TestFakeEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return DeterministicFakeEmbedding

    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return {"size": 6}

from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from typing_extensions import override

from langchain_tests.integration_tests import EmbeddingsIntegrationTests
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestFakeEmbeddingsUnit(EmbeddingsUnitTests):
    @override
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return DeterministicFakeEmbedding

    @override
    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return {"size": 6}  # embedding dimension


class TestFakeEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @override
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return DeterministicFakeEmbedding

    @override
    @property
    def embedding_model_params(self) -> dict[str, Any]:
        return {"size": 6}

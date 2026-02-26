"""Test __ModuleName__ embeddings."""

from typing import Type

from __module_name__.embeddings import __ModuleName__Embeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[__ModuleName__Embeddings]:
        return __ModuleName__Embeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}

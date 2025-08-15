"""Test embedding model integration."""

from typing import Type

from __module_name__.embeddings import __ModuleName__Embeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestParrotLinkEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[__ModuleName__Embeddings]:
        return __ModuleName__Embeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}

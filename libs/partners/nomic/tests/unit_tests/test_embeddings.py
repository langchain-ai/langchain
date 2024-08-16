"""Test embedding model integration."""


from langchain_nomic.embeddings import NomicEmbeddings

from langchain_standard_tests.unit_tests import EmbeddingsUnitTests
from typing import Type

class TestNomicEmbeddings(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[NomicEmbeddings]:
        return NomicEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nomic-embed-text-v1"}

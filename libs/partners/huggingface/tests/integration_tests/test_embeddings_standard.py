"""Test HuggingFace embeddings."""

from typing import Type

from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_huggingface.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpointEmbeddings,
)


class TestHuggingFaceEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[HuggingFaceEmbeddings]:
        return HuggingFaceEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model_name": "sentence-transformers/all-mpnet-base-v2"}


class TestHuggingFaceEndpointEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[HuggingFaceEndpointEmbeddings]:
        return HuggingFaceEndpointEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "sentence-transformers/all-mpnet-base-v2"}

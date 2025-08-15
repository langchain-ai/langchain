"""Test HuggingFace embeddings."""

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_huggingface.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpointEmbeddings,
)


class TestHuggingFaceEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[HuggingFaceEmbeddings]:
        return HuggingFaceEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model_name": "sentence-transformers/all-mpnet-base-v2"}


class TestHuggingFaceEndpointEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[HuggingFaceEndpointEmbeddings]:
        return HuggingFaceEndpointEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "sentence-transformers/all-mpnet-base-v2"}

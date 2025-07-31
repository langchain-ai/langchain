"""Test HuggingFace embeddings."""

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_huggingface.embeddings import (
    HuggingFaceEndpointEmbeddings,
)


class TestHuggingFaceEndpointEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[HuggingFaceEndpointEmbeddings]:
        return HuggingFaceEndpointEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "microsoft/all-mpnet-base-v2"}

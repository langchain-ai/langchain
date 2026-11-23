"""Standard integration tests for `PerplexityEmbeddings`."""

import os

import pytest
from langchain_core.embeddings import Embeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_perplexity import PerplexityEmbeddings


@pytest.mark.skipif(
    not (os.environ.get("PPLX_API_KEY") or os.environ.get("PERPLEXITY_API_KEY")),
    reason="PPLX_API_KEY/PERPLEXITY_API_KEY not set",
)
class TestPerplexityEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return PerplexityEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {}

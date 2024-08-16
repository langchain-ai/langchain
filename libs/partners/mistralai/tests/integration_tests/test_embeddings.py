"""Test MistralAI Embedding"""

from langchain_mistralai import MistralAIEmbeddings

from langchain_standard_tests.integration_tests import EmbeddingsIntegrationTests
from typing import Type

class TestOllamaEmbeddings(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[MistralAIEmbeddings]:
        return MistralAIEmbeddings
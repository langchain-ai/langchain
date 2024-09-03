from abc import abstractmethod
from typing import Type

import pytest
from langchain_core.embeddings import Embeddings

from langchain_standard_tests.base import BaseStandardTests


class EmbeddingsTests(BaseStandardTests):
    @property
    @abstractmethod
    def embeddings_class(self) -> Type[Embeddings]:
        ...

    @property
    def embedding_model_params(self) -> dict:
        return {}

    @pytest.fixture
    def model(self) -> Embeddings:
        return self.embeddings_class(**self.embedding_model_params)


class EmbeddingsUnitTests(EmbeddingsTests):
    def test_init(self) -> None:
        model = self.embeddings_class(**self.embedding_model_params)
        assert model is not None

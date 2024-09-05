import os
from abc import abstractmethod
from typing import Tuple, Type
from unittest import mock

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import SecretStr

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

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Return env vars, init args, and expected instance attrs for initializing
        from env vars."""
        return {}, {}, {}

    def test_init_from_env(self) -> None:
        env_params, embeddings_params, expected_attrs = self.init_from_env_params
        if env_params:
            with mock.patch.dict(os.environ, env_params):
                model = self.embeddings_class(**embeddings_params)
            assert model is not None
            for k, expected in expected_attrs.items():
                actual = getattr(model, k)
                if isinstance(actual, SecretStr):
                    actual = actual.get_secret_value()
                assert actual == expected

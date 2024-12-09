import os
from abc import abstractmethod
from typing import Tuple, Type
from unittest import mock

import pytest
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from langchain_tests.base import BaseStandardTests


class EmbeddingsTests(BaseStandardTests):
    """
    :private:
    """

    @property
    @abstractmethod
    def embeddings_class(self) -> Type[Embeddings]: ...

    @property
    def embedding_model_params(self) -> dict:
        return {}

    @pytest.fixture
    def model(self) -> Embeddings:
        return self.embeddings_class(**self.embedding_model_params)


class EmbeddingsUnitTests(EmbeddingsTests):
    """Base class for embeddings unit tests.

    Test subclasses must implement the ``embeddings_class`` property to specify the
    embeddings model to be tested. You can also override the
    ``embedding_model_params`` property to specify initialization parameters.

    Example:

    .. code-block:: python

        from typing import Type

        from langchain_tests.unit_tests import EmbeddingsUnitTests
        from my_package.embeddings import MyEmbeddingsModel


        class TestMyEmbeddingsModelUnit(EmbeddingsUnitTests):
            @property
            def embeddings_class(self) -> Type[MyEmbeddingsModel]:
                # Return the embeddings model class to test here
                return MyEmbeddingsModel

            @property
            def embedding_model_params(self) -> dict:
                # Return initialization parameters for the model.
                return {"model": "model-001"}

    .. note::
          API references for individual test methods include troubleshooting tips.

    Testing initialization from environment variables
        Overriding the ``init_from_env_params`` property will enable additional tests
        for initialization from environment variables. See below for details.

        .. dropdown:: init_from_env_params

            This property is used in unit tests to test initialization from
            environment variables. It should return a tuple of three dictionaries
            that specify the environment variables, additional initialization args,
            and expected instance attributes to check.

            Defaults to empty dicts. If not overridden, the test is skipped.

            Example:

            .. code-block:: python

                @property
                def init_from_env_params(self) -> Tuple[dict, dict, dict]:
                    return (
                        {
                            "MY_API_KEY": "api_key",
                        },
                        {
                            "model": "model-001",
                        },
                        {
                            "my_api_key": "api_key",
                        },
                    )
    """  # noqa: E501

    def test_init(self) -> None:
        """Test model initialization.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that ``embedding_model_params`` is specified
            and the model can be initialized from those params.
        """
        model = self.embeddings_class(**self.embedding_model_params)
        assert model is not None

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """This property is used in unit tests to test initialization from environment
        variables. It should return a tuple of three dictionaries that specify the
        environment variables, additional initialization args, and expected instance
        attributes to check."""
        return {}, {}, {}

    def test_init_from_env(self) -> None:
        """Test initialization from environment variables. Relies on the
        ``init_from_env_params`` property. Test is skipped if that property is not
        set.

        .. dropdown:: Troubleshooting

            If this test fails, ensure that ``init_from_env_params`` is specified
            correctly and that model parameters are properly set from environment
            variables during initialization.
        """
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

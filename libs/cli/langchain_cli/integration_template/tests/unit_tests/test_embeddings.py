"""Test embedding model integration."""

from typing import Tuple, Type

from langchain_core.embeddings import Embeddings
from langchain_standard_tests.unit_tests.embeddings import EmbeddingsUnitTests

from __module_name__.embeddings import __ModuleName__Embeddings


class TestFireworksStandard(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[Embeddings]:
        return __ModuleName__Embeddings

    @property
    def embeddings_params(self) -> dict:
        return {"api_key": "test api key"}

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Return env vars, init args, and expected instance attrs for initializing
        from env vars.

        This powers tests for initializing from environment variables."""
        return (
            # env vars
            {
                "__MODULE_NAME___API_KEY": "test api key",
            },
            # init vars - only pass things that are required and CAN'T be set
            # via env vars
            {},
            # expected attributes once the object has been constructed
            {
                "api_key": "test api key",
            },
        )


def test_initialization() -> None:
    """Test embedding model initialization."""
    __ModuleName__Embeddings()

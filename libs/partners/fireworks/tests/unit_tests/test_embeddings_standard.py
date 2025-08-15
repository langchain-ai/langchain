"""Standard LangChain interface tests."""

from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests.embeddings import EmbeddingsUnitTests

from langchain_fireworks import FireworksEmbeddings


class TestFireworksStandard(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return FireworksEmbeddings

    @property
    def embeddings_params(self) -> dict:
        return {"api_key": "test_api_key"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "FIREWORKS_API_KEY": "api_key",
            },
            {},
            {
                "fireworks_api_key": "api_key",
            },
        )

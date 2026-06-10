"""Standard unit tests for `PerplexityEmbeddings`."""

from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_perplexity import PerplexityEmbeddings


class TestPerplexityEmbeddingsStandard(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[Embeddings]:
        return PerplexityEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"pplx_api_key": "test"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return ({"PPLX_API_KEY": "api_key"}, {}, {"pplx_api_key": "api_key"})

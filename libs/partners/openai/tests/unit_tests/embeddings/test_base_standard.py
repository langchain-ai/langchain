"""Standard LangChain interface tests"""

from typing import Tuple, Type

from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests.embeddings import EmbeddingsUnitTests

from langchain_openai import OpenAIEmbeddings


class TestOpenAIStandard(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[Embeddings]:
        return OpenAIEmbeddings

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {
                "OPENAI_API_KEY": "api_key",
                "OPENAI_ORG_ID": "org_id",
                "OPENAI_API_BASE": "api_base",
                "OPENAI_PROXY": "https://proxy.com",
            },
            {},
            {
                "openai_api_key": "api_key",
                "openai_organization": "org_id",
                "openai_api_base": "api_base",
                "openai_proxy": "https://proxy.com",
            },
        )

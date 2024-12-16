"""Standard LangChain interface tests"""

from typing import Tuple, Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_openai import ChatOpenAI


class TestOpenAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAI

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

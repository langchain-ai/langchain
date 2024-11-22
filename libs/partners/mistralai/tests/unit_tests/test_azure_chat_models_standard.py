"""Standard LangChain interface tests"""

from typing import Tuple, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_mistralai import AzureChatMistralAI


class TestAzureMistralAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return AzureChatMistralAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "azure_endpoint": "https://test.azure.com",
        }

    @pytest.mark.xfail(reason="Azure does not support tool_choice='any'")
    def test_bind_tool_pydantic(self, model: BaseChatModel) -> None:
        super().test_bind_tool_pydantic(model)

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {
                "AZURE_MISTRAL_ENDPOINT": "https://azure.mistral-endpoint.com/",
                "AZURE_MISTRAL_API_KEY": "api_key",
            },
            {},
            {
                "mistral_api_key": "api_key",
                "azure_endpoint": "https://azure.mistral-endpoint.com/",
            },
        )

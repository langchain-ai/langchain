"""Standard LangChain interface tests"""

from typing import Tuple, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_openai import AzureChatOpenAI


class TestOpenAIStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return AzureChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "deployment_name": "test",
            "openai_api_version": "2021-10-01",
            "azure_endpoint": "https://test.azure.com",
        }

    @pytest.mark.xfail(reason="AzureOpenAI does not support tool_choice='any'")
    def test_bind_tool_pydantic(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_bind_tool_pydantic(model, my_adder_tool)

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return (
            {
                "AZURE_OPENAI_API_KEY": "api_key",
                "AZURE_OPENAI_ENDPOINT": "https://endpoint.com",
                "AZURE_OPENAI_AD_TOKEN": "token",
                "OPENAI_ORG_ID": "org_id",
                "OPENAI_API_VERSION": "yyyy-mm-dd",
                "OPENAI_API_TYPE": "type",
            },
            {},
            {
                "openai_api_key": "api_key",
                "azure_endpoint": "https://endpoint.com",
                "azure_ad_token": "token",
                "openai_organization": "org_id",
                "openai_api_version": "yyyy-mm-dd",
                "openai_api_type": "type",
            },
        )

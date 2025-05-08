"""Standard LangChain interface tests"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_openai import AzureChatOpenAI

OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "")
OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "")


class TestAzureOpenAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return AzureChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "deployment_name": os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            "model": "gpt-4o-mini",
            "openai_api_version": OPENAI_API_VERSION,
            "azure_endpoint": OPENAI_API_BASE,
            "stream_usage": True,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True


class TestAzureOpenAIStandardLegacy(ChatModelIntegrationTests):
    """Test a legacy model."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return AzureChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "deployment_name": os.environ["AZURE_OPENAI_LEGACY_CHAT_DEPLOYMENT_NAME"],
            "openai_api_version": OPENAI_API_VERSION,
            "azure_endpoint": OPENAI_API_BASE,
            "stream_usage": True,
        }

    @property
    def structured_output_kwargs(self) -> dict:
        return {"method": "function_calling"}

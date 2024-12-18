"""Standard LangChain interface tests"""

import os
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_openai import AzureChatOpenAI

OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "")
OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "")


class TestAzureOpenAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return AzureChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "deployment_name": os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            "model": "gpt-4o",
            "openai_api_version": OPENAI_API_VERSION,
            "azure_endpoint": OPENAI_API_BASE,
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @pytest.mark.xfail(reason="Not yet supported.")
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)


class TestAzureOpenAIStandardLegacy(ChatModelIntegrationTests):
    """Test a legacy model."""

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return AzureChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "deployment_name": os.environ["AZURE_OPENAI_LEGACY_CHAT_DEPLOYMENT_NAME"],
            "openai_api_version": OPENAI_API_VERSION,
            "azure_endpoint": OPENAI_API_BASE,
        }

    @pytest.mark.xfail(reason="Not yet supported.")
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

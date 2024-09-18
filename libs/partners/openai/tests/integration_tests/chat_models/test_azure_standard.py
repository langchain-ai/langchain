"""Standard LangChain interface tests"""

import os
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_openai import AzureChatOpenAI


class TestAzureOpenAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return AzureChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "deployment_name": os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            "model": "gpt-4o",
        }

    @property
    def supports_image_inputs(self) -> bool:
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
            "deployment_name": os.environ["AZURE_OPENAI_LEGACY_CHAT_DEPLOYMENT_NAME"]
        }

    @pytest.mark.xfail(reason="Not yet supported.")
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

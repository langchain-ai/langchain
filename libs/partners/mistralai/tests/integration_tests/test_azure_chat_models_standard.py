"""Standard LangChain interface tests"""

import os
from typing import Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_mistralai.chat_models.azure import AzureChatMistralAI

AZURE_MISTRAL_ENDPOINT = os.environ.get("AZURE_MISTRAL_ENDPOINT", "")
AZURE_MISTRAL_API_KEY = os.environ.get("AZURE_MISTRAL_API_KEY", "")


class TestAzureMistralAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return AzureChatMistralAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "azure_endpoint": AZURE_MISTRAL_ENDPOINT,
            "api_key": AZURE_MISTRAL_API_KEY,
        }

    @property
    def tool_choice_value(self) -> Optional[str]:
        """Value to use for tool choice when used in tests."""
        return "any"

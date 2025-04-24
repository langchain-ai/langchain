"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_community.chat_models.litellm import ChatLiteLLM


class TestLiteLLMStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatLiteLLM

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "ollama/mistral",
            # Needed to get the usage object when streaming. See https://docs.litellm.ai/docs/completion/usage#streaming-usage
            "model_kwargs": {"stream_options": {"include_usage": True}},
        }

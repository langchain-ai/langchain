"""Test chat model integration."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_ollama.chat_models import ChatOllama


class TestChatOllama(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatOllama]:
        return ChatOllama

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama3-groq-tool-use"}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason=(
            "Fails with 'AssertionError'. Ollama does not support 'tool_choice' yet."
        )
    )
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(
        reason=(
            "Fails with 'AssertionError'. Ollama does not support 'tool_choice' yet."
        )
    )
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

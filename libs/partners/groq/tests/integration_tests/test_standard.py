"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_groq import ChatGroq


class BaseTestGroq(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGroq

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_list_content(model)


class TestGroqMixtral(BaseTestGroq):
    @property
    def chat_model_params(self) -> dict:
        return {
            "temperature": 0,
        }

    @pytest.mark.xfail(
        reason=("Fails with 'Failed to call a function. Please adjust your prompt.'")
    )
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(
        reason=("May pass arguments: {'properties': {}, 'type': 'object'}")
    )
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)


class TestGroqLlama(BaseTestGroq):
    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "llama3-8b-8192",
            "temperature": 0,
        }

    @pytest.mark.xfail(
        reason=("Fails with 'Failed to call a function. Please adjust your prompt.'")
    )
    def test_tool_message_histories_string_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_string_content(model)

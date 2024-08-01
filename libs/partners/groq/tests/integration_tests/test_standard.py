"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_standard_tests.integration_tests import (
    ChatModelIntegrationTests,
)

from langchain_groq import ChatGroq

rate_limiter = InMemoryRateLimiter(requests_per_second=0.45)


class BaseTestGroq(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGroq

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_list_content(model)


class TestGroqLlama(BaseTestGroq):
    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "llama-3.1-8b-instant",
            "temperature": 0,
            "rate_limiter": rate_limiter,
        }

    @pytest.mark.xfail(
        reason=("Fails with 'Failed to call a function. Please adjust your prompt.'")
    )
    def test_tool_message_histories_string_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_string_content(model)

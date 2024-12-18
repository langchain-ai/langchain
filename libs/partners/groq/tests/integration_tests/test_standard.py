"""Standard LangChain interface tests"""

from typing import Optional, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import (
    ChatModelIntegrationTests,
)

from langchain_groq import ChatGroq

rate_limiter = InMemoryRateLimiter(requests_per_second=0.2)


class BaseTestGroq(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatGroq

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)

    @property
    def supports_json_mode(self) -> bool:
        return True


class TestGroqLlama(BaseTestGroq):
    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "llama-3.1-8b-instant",
            "temperature": 0,
            "rate_limiter": rate_limiter,
        }

    @property
    def tool_choice_value(self) -> Optional[str]:
        """Value to use for tool choice when used in tests."""
        return "any"

    @property
    def supports_json_mode(self) -> bool:
        return False  # Not supported in streaming mode

    @pytest.mark.xfail(
        reason=("Fails with 'Failed to call a function. Please adjust your prompt.'")
    )
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)

    @pytest.mark.xfail(
        reason=("Fails with 'Failed to call a function. Please adjust your prompt.'")
    )
    def test_tool_message_histories_string_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_string_content(model, my_adder_tool)

    @pytest.mark.xfail(
        reason=(
            "Sometimes fails with 'Failed to call a function. "
            "Please adjust your prompt.'"
        )
    )
    def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
        super().test_bind_runnables_as_tools(model)

"""Standard LangChain interface tests"""

from typing import Optional, Type

import pytest  # type: ignore[import-not-found]
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_xai import ChatXAI

# Initialize the rate limiter in global scope, so it can be re-used
# across tests.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
)


class TestXAIStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatXAI

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "grok-beta",
            "rate_limiter": rate_limiter,
        }

    @property
    def tool_choice_value(self) -> Optional[str]:
        """Value to use for tool choice when used in tests."""
        return "tool_name"

    @pytest.mark.xfail(reason="Not yet supported.")
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)

    @pytest.mark.xfail(reason="Can't handle AIMessage with empty content.")
    def test_tool_message_error_status(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_error_status(model, my_adder_tool)

    @pytest.mark.xfail(reason="Can't handle AIMessage with empty content.")
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_structured_few_shot_examples(model, my_adder_tool)

    @pytest.mark.xfail(reason="Can't handle AIMessage with empty content.")
    def test_tool_message_histories_string_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_string_content(model, my_adder_tool)

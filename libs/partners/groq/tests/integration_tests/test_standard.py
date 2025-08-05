"""Standard LangChain interface tests."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import (
    ChatModelIntegrationTests,
)

from langchain_groq import ChatGroq

rate_limiter = InMemoryRateLimiter(requests_per_second=0.2)


class TestGroq(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatGroq

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama-3.3-70b-versatile", "rate_limiter": rate_limiter}

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)

    @property
    def supports_json_mode(self) -> bool:
        return True

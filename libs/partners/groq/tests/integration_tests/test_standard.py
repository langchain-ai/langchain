"""Standard LangChain interface tests."""

from typing import Literal

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

    @pytest.mark.xfail(
        reason="Groq models have inconsistent tool calling performance. See: "
        "https://github.com/langchain-ai/langchain/discussions/19990"
    )
    def test_bind_runnables_as_tools(self, model: BaseChatModel) -> None:
        super().test_bind_runnables_as_tools(model)

    @property
    def supports_json_mode(self) -> bool:
        return True


@pytest.mark.parametrize("schema_type", ["pydantic", "typeddict", "json_schema"])
def test_json_schema(
    schema_type: Literal["pydantic", "typeddict", "json_schema"],
) -> None:
    class JsonSchemaTests(ChatModelIntegrationTests):
        @property
        def chat_model_class(self) -> type[ChatGroq]:
            return ChatGroq

        @property
        def chat_model_params(self) -> dict:
            return {"model": "openai/gpt-oss-120b", "rate_limiter": rate_limiter}

        @property
        def structured_output_kwargs(self) -> dict:
            return {"method": "json_schema"}

    test_instance = JsonSchemaTests()
    model = test_instance.chat_model_class(**test_instance.chat_model_params)
    JsonSchemaTests().test_structured_output(model, schema_type)

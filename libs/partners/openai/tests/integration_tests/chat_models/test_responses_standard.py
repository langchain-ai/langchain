"""Standard LangChain interface tests for Responses API"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from langchain_openai import ChatOpenAI
from tests.integration_tests.chat_models.test_base_standard import TestOpenAIStandard


class TestOpenAIResponses(TestOpenAIStandard):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-4o-mini", "use_responses_api": True}

    @pytest.mark.xfail(reason="Unsupported.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)

    # The below tests are failing because they all feature AIMessages of the form
    #
    #    AIMessage(
    #        content=...,
    #        tool_calls=[
    #            {
    #                "name": function_name,
    #                "args": function_args,
    #                "id": "abc123",
    #                "type": "tool_call",
    #            },
    #        ],
    #    )
    #
    # In addition to tool call IDs, the Respones API requires an ID prefixed by "fc_*"
    # that identifies the originating "function_call" output block.

    @pytest.mark.xfail(reason="TODO: Support.")
    def test_tool_message_histories_string_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_string_content(model, my_adder_tool)

    @pytest.mark.xfail(reason="TODO: Support.")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)

    @pytest.mark.xfail(reason="TODO: Support.")
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_structured_few_shot_examples(model, my_adder_tool)

    @pytest.mark.xfail(reason="TODO: Support.")
    def test_anthropic_inputs(self, model: BaseChatModel) -> None:
        super().test_anthropic_inputs(model)

    @pytest.mark.xfail(reason="TODO: Support.")
    def test_tool_message_error_status(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_error_status(model, my_adder_tool)

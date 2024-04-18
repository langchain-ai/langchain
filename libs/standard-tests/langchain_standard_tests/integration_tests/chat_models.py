import json
from abc import ABC, abstractmethod
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool


class Person(BaseModel):
    name: str = Field(..., description="The name of the person.")
    age: int = Field(..., description="The age of the person.")


@tool
def my_adder_tool(a: int, b: int) -> int:
    """Takes two integers, a and b, and returns their sum."""
    return a + b


class ChatModelIntegrationTests(ABC):
    @abstractmethod
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        ...

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {}

    @pytest.fixture
    def chat_model_has_tool_calling(
        self, chat_model_class: Type[BaseChatModel]
    ) -> bool:
        return chat_model_class.bind_tools is not BaseChatModel.bind_tools

    @pytest.fixture
    def chat_model_has_structured_output(
        self, chat_model_class: Type[BaseChatModel]
    ) -> bool:
        return (
            chat_model_class.with_structured_output
            is not BaseChatModel.with_structured_output
        )

    def test_invoke(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        model = chat_model_class(**chat_model_params)
        result = model.invoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    async def test_ainvoke(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        model = chat_model_class(**chat_model_params)
        result = await model.ainvoke("Hello")
        assert result is not None
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)
        assert len(result.content) > 0

    def test_stream(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        model = chat_model_class(**chat_model_params)
        num_tokens = 0
        for token in model.stream("Hello"):
            assert token is not None
            assert isinstance(token, AIMessageChunk)
            assert isinstance(token.content, str)
            num_tokens += len(token.content)
        assert num_tokens > 0

    async def test_astream(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        model = chat_model_class(**chat_model_params)
        num_tokens = 0
        async for token in model.astream("Hello"):
            assert token is not None
            assert isinstance(token, AIMessageChunk)
            assert isinstance(token.content, str)
            num_tokens += len(token.content)
        assert num_tokens > 0

    def test_batch(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        model = chat_model_class(**chat_model_params)
        batch_results = model.batch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert isinstance(result.content, str)
            assert len(result.content) > 0

    async def test_abatch(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        model = chat_model_class(**chat_model_params)
        batch_results = await model.abatch(["Hello", "Hey"])
        assert batch_results is not None
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        for result in batch_results:
            assert result is not None
            assert isinstance(result, AIMessage)
            assert isinstance(result.content, str)
            assert len(result.content) > 0

    def test_tool_message_histories(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
        chat_model_has_tool_calling: bool,
    ) -> None:
        """Test that message histories are compatible across providers."""
        if not chat_model_has_tool_calling:
            pytest.skip("Test requires tool calling.")
        model = chat_model_class(**chat_model_params)
        model_with_tools = model.bind_tools([my_adder_tool])
        function_name = "my_adder_tool"
        function_args = {"a": "1", "b": "2"}

        human_message = HumanMessage(content="What is 1 + 2")
        tool_message = ToolMessage(
            name=function_name,
            content=json.dumps({"result": 3}),
            tool_call_id="abc123",
        )

        # String content (e.g., OpenAI)
        string_content_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": function_name,
                    "args": function_args,
                    "id": "abc123",
                },
            ],
        )
        messages = [
            human_message,
            string_content_msg,
            tool_message,
        ]
        result = model_with_tools.invoke(messages)
        assert isinstance(result, AIMessage)

        # List content (e.g., Anthropic)
        list_content_msg = AIMessage(
            content=[
                {"type": "text", "text": "some text"},
                {
                    "type": "tool_use",
                    "id": "abc123",
                    "name": function_name,
                    "input": function_args,
                },
            ],
            tool_calls=[
                {
                    "name": function_name,
                    "args": function_args,
                    "id": "abc123",
                },
            ],
        )
        messages = [
            human_message,
            list_content_msg,
            tool_message,
        ]
        result = model_with_tools.invoke(messages)
        assert isinstance(result, AIMessage)

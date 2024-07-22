"""Test ChatDeepInfra wrapper."""

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import RunnableBinding

from langchain_community.chat_models.deepinfra import ChatDeepInfra
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


class GenerateMovieName(BaseModel):
    "Get a movie name from a description"

    description: str


def test_chat_deepinfra() -> None:
    """Test valid call to DeepInfra."""
    chat = ChatDeepInfra(
        max_tokens=10,
    )
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_deepinfra_streaming() -> None:
    callback_handler = FakeCallbackHandler()
    chat = ChatDeepInfra(
        callbacks=[callback_handler],
        streaming=True,
        max_tokens=10,
    )
    response = chat.invoke([HumanMessage(content="Hello")])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


async def test_async_chat_deepinfra() -> None:
    """Test async generation."""
    chat = ChatDeepInfra(
        max_tokens=10,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert len(response.generations[0]) == 1
    generation = response.generations[0][0]
    assert isinstance(generation, ChatGeneration)
    assert isinstance(generation.text, str)
    assert generation.text == generation.message.content


async def test_async_chat_deepinfra_streaming() -> None:
    callback_handler = FakeCallbackHandler()
    chat = ChatDeepInfra(
        # model="meta-llama/Llama-2-7b-chat-hf",
        callbacks=[callback_handler],
        max_tokens=10,
        streaming=True,
        timeout=5,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert len(response.generations[0]) == 1
    generation = response.generations[0][0]
    assert isinstance(generation, ChatGeneration)
    assert isinstance(generation.text, str)
    assert generation.text == generation.message.content


def test_chat_deepinfra_bind_tools() -> None:
    class Foo(BaseModel):
        pass

    chat = ChatDeepInfra(
        max_tokens=10,
    )
    tools = [Foo]
    chat_with_tools = chat.bind_tools(tools)
    assert isinstance(chat_with_tools, RunnableBinding)
    chat_tools = chat_with_tools.tools
    assert chat_tools
    assert chat_tools == {
        "tools": [
            {
                "function": {
                    "description": "",
                    "name": "Foo",
                    "parameters": {"properties": {}, "type": "object"},
                },
                "type": "function",
            }
        ]
    }


def test_tool_use() -> None:
    llm = ChatDeepInfra(model="meta-llama/Meta-Llama-3-70B-Instruct", temperature=0)
    llm_with_tool = llm.bind_tools(tools=[GenerateMovieName], tool_choice=True)
    msgs: List = [
        HumanMessage(content="It should be a movie explaining humanity in 2133.")
    ]
    ai_msg = llm_with_tool.invoke(msgs)

    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call

    tool_msg = ToolMessage(
        content="Year 2133",
        tool_call_id=ai_msg.additional_kwargs["tool_calls"][0]["id"],
    )
    msgs.extend([ai_msg, tool_msg])
    llm_with_tool.invoke(msgs)

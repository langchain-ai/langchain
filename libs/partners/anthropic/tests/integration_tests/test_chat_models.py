"""Test ChatAnthropic chat model."""

from __future__ import annotations

import asyncio
import json
import os
from base64 import b64encode
from typing import Literal, cast

import httpx
import pytest
import requests
from anthropic import BadRequestError
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.callbacks import CallbackManager
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_anthropic._compat import _convert_from_v1_to_anthropic
from tests.unit_tests._utils import FakeCallbackHandler

MODEL_NAME = "claude-3-5-haiku-20241022"


def test_stream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    full: BaseMessageChunk | None = None
    chunks_with_input_token_counts = 0
    chunks_with_output_token_counts = 0
    chunks_with_model_name = 0
    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)
        full = cast("BaseMessageChunk", token) if full is None else full + token
        assert isinstance(token, AIMessageChunk)
        if token.usage_metadata is not None:
            if token.usage_metadata.get("input_tokens"):
                chunks_with_input_token_counts += 1
            if token.usage_metadata.get("output_tokens"):
                chunks_with_output_token_counts += 1
        chunks_with_model_name += int("model_name" in token.response_metadata)
    if chunks_with_input_token_counts != 1 or chunks_with_output_token_counts != 1:
        msg = (
            "Expected exactly one chunk with input or output token counts. "
            "AIMessageChunk aggregation adds counts. Check that "
            "this is behaving properly."
        )
        raise AssertionError(
            msg,
        )
    assert chunks_with_model_name == 1
    # check token usage is populated
    assert isinstance(full, AIMessageChunk)
    assert len(full.content_blocks) == 1
    assert full.content_blocks[0]["type"] == "text"
    assert full.content_blocks[0]["text"]
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert full.usage_metadata["total_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert "stop_reason" in full.response_metadata
    assert "stop_sequence" in full.response_metadata
    assert "model_name" in full.response_metadata


async def test_astream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    full: BaseMessageChunk | None = None
    chunks_with_input_token_counts = 0
    chunks_with_output_token_counts = 0
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)
        full = cast("BaseMessageChunk", token) if full is None else full + token
        assert isinstance(token, AIMessageChunk)
        if token.usage_metadata is not None:
            if token.usage_metadata.get("input_tokens"):
                chunks_with_input_token_counts += 1
            if token.usage_metadata.get("output_tokens"):
                chunks_with_output_token_counts += 1
    if chunks_with_input_token_counts != 1 or chunks_with_output_token_counts != 1:
        msg = (
            "Expected exactly one chunk with input or output token counts. "
            "AIMessageChunk aggregation adds counts. Check that "
            "this is behaving properly."
        )
        raise AssertionError(
            msg,
        )
    # check token usage is populated
    assert isinstance(full, AIMessageChunk)
    assert len(full.content_blocks) == 1
    assert full.content_blocks[0]["type"] == "text"
    assert full.content_blocks[0]["text"]
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert full.usage_metadata["total_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    assert "stop_reason" in full.response_metadata
    assert "stop_sequence" in full.response_metadata

    # Check expected raw API output
    async_client = llm._async_client
    params: dict = {
        "model": MODEL_NAME,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }
    stream = await async_client.messages.create(**params, stream=True)
    async for event in stream:
        if event.type == "message_start":
            assert event.message.usage.input_tokens > 1
            # Different models may report different initial output token counts
            # in the message_start event. Ensure it's a positive value.
            assert event.message.usage.output_tokens >= 1
        elif event.type == "message_delta":
            assert event.usage.output_tokens >= 1
        else:
            pass


async def test_stream_usage() -> None:
    """Test usage metadata can be excluded."""
    model = ChatAnthropic(model_name=MODEL_NAME, stream_usage=False)  # type: ignore[call-arg]
    async for token in model.astream("hi"):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None


async def test_stream_usage_override() -> None:
    # check we override with kwarg
    model = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg]
    assert model.stream_usage
    async for token in model.astream("hi", stream_usage=False):
        assert isinstance(token, AIMessageChunk)
        assert token.usage_metadata is None


async def test_abatch() -> None:
    """Test streaming tokens."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"],
        config={"tags": ["foo"]},
    )
    for token in result:
        assert isinstance(token.content, str)


async def test_async_tool_use() -> None:
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )

    llm_with_tools = llm.bind_tools(
        [
            {
                "name": "get_weather",
                "description": "Get weather report for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        ],
    )
    response = await llm_with_tools.ainvoke("what's the weather in san francisco, ca")
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]

    # Test streaming
    first = True
    chunks: list[BaseMessage | BaseMessageChunk] = []
    async for chunk in llm_with_tools.astream(
        "what's the weather in san francisco, ca",
    ):
        chunks = [*chunks, chunk]
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore[assignment]
    assert len(chunks) > 1
    assert isinstance(gathered, AIMessageChunk)
    assert isinstance(gathered.tool_call_chunks, list)
    assert len(gathered.tool_call_chunks) == 1
    tool_call_chunk = gathered.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "get_weather"
    assert isinstance(tool_call_chunk["args"], str)
    assert "location" in json.loads(tool_call_chunk["args"])


def test_batch() -> None:
    """Test batch tokens."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)
    assert "model_name" in result.response_metadata


def test_invoke() -> None:
    """Test invoke tokens."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = llm.invoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_system_invoke() -> None:
    """Test invoke tokens with a system message."""
    llm = ChatAnthropic(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert cartographer. If asked, you are a cartographer. "
                "STAY IN CHARACTER",
            ),
            ("human", "Are you a mathematician?"),
        ],
    )

    chain = prompt | llm

    result = chain.invoke({})
    assert isinstance(result.content, str)


def test_handle_empty_aimessage() -> None:
    # Anthropic can generate empty AIMessages, which are not valid unless in the last
    # message in a sequence.
    llm = ChatAnthropic(model=MODEL_NAME)
    messages = [
        HumanMessage("Hello"),
        AIMessage([]),
        HumanMessage("My name is Bob."),
    ]
    _ = llm.invoke(messages)

    # Test tool call sequence
    llm_with_tools = llm.bind_tools(
        [
            {
                "name": "get_weather",
                "description": "Get weather report for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        ],
    )
    _ = llm_with_tools.invoke(
        [
            HumanMessage("What's the weather in Boston?"),
            AIMessage(
                content=[],
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"location": "Boston"},
                        "id": "toolu_01V6d6W32QGGSmQm4BT98EKk",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                content="It's sunny.", tool_call_id="toolu_01V6d6W32QGGSmQm4BT98EKk"
            ),
            AIMessage([]),
            HumanMessage("Thanks!"),
        ]
    )


def test_anthropic_call() -> None:
    """Test valid call to anthropic."""
    chat = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_anthropic_generate() -> None:
    """Test generate method of anthropic."""
    chat = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    chat_messages: list[list[BaseMessage]] = [
        [HumanMessage(content="How many toes do dogs have?")],
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy


def test_anthropic_streaming() -> None:
    """Test streaming tokens from anthropic."""
    chat = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.stream([message])
    for token in response:
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)


def test_anthropic_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        callbacks=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    for token in chat.stream([message]):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
    assert callback_handler.llm_streams > 1


async def test_anthropic_async_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        callbacks=callback_manager,
        verbose=True,
    )
    chat_messages: list[BaseMessage] = [
        HumanMessage(content="How many toes do dogs have?"),
    ]
    async for token in chat.astream(chat_messages):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
    assert callback_handler.llm_streams > 1


def test_anthropic_multimodal() -> None:
    """Test that multimodal inputs are handled correctly."""
    chat = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    messages: list[BaseMessage] = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        # langchain logo
                        "url": "data:image/jpeg;base64,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",  # noqa: E501
                    },
                },
                {"type": "text", "text": "What is this a logo for?"},
            ],
        ),
    ]
    response = chat.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    num_tokens = chat.get_num_tokens_from_messages(messages)
    assert num_tokens > 0


def test_streaming() -> None:
    """Test streaming tokens from Anthropic."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])

    llm = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model_name=MODEL_NAME,
        streaming=True,
        callbacks=callback_manager,
    )

    response = llm.generate([[HumanMessage(content="I'm Pickle Rick")]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)


async def test_astreaming() -> None:
    """Test streaming tokens from Anthropic."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])

    llm = ChatAnthropic(  # type: ignore[call-arg, call-arg]
        model_name=MODEL_NAME,
        streaming=True,
        callbacks=callback_manager,
    )

    response = await llm.agenerate([[HumanMessage(content="I'm Pickle Rick")]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)


def test_tool_use() -> None:
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",  # type: ignore[call-arg]
        temperature=0,
    )
    tool_definition = {
        "name": "get_weather",
        "description": "Get weather report for a city",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    }
    llm_with_tools = llm.bind_tools([tool_definition])
    query = "how are you? what's the weather in san francisco, ca"
    response = llm_with_tools.invoke(query)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]

    content_blocks = response.content_blocks
    assert len(content_blocks) == 2
    assert content_blocks[0]["type"] == "text"
    assert content_blocks[0]["text"]
    assert content_blocks[1]["type"] == "tool_call"
    assert content_blocks[1]["name"] == "get_weather"
    assert content_blocks[1]["args"] == tool_call["args"]

    # Test streaming
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",  # type: ignore[call-arg]
        temperature=0,
        # Add extra headers to also test token-efficient tools
        model_kwargs={
            "extra_headers": {"anthropic-beta": "token-efficient-tools-2025-02-19"},
        },
    )
    llm_with_tools = llm.bind_tools([tool_definition])
    first = True
    chunks: list[BaseMessage | BaseMessageChunk] = []
    for chunk in llm_with_tools.stream(query):
        chunks = [*chunks, chunk]
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore[assignment]
        for block in chunk.content_blocks:
            assert block["type"] in ("text", "tool_call_chunk")
    assert len(chunks) > 1
    assert isinstance(gathered.content, list)
    assert len(gathered.content) == 2
    tool_use_block = None
    for content_block in gathered.content:
        assert isinstance(content_block, dict)
        if content_block["type"] == "tool_use":
            tool_use_block = content_block
            break
    assert tool_use_block is not None
    assert tool_use_block["name"] == "get_weather"
    assert "location" in json.loads(tool_use_block["partial_json"])
    assert isinstance(gathered, AIMessageChunk)
    assert isinstance(gathered.tool_calls, list)
    assert len(gathered.tool_calls) == 1
    tool_call = gathered.tool_calls[0]
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]
    assert tool_call["id"] is not None

    content_blocks = gathered.content_blocks
    assert len(content_blocks) == 2
    assert content_blocks[0]["type"] == "text"
    assert content_blocks[0]["text"]
    assert content_blocks[1]["type"] == "tool_call"
    assert content_blocks[1]["name"] == "get_weather"
    assert content_blocks[1]["args"]

    # Testing token-efficient tools
    # https://platform.claude.com/docs/en/agents-and-tools/tool-use/token-efficient-tool-use
    assert gathered.usage_metadata
    assert response.usage_metadata
    assert (
        gathered.usage_metadata["total_tokens"]
        < response.usage_metadata["total_tokens"]
    )

    # Test passing response back to model
    stream = llm_with_tools.stream(
        [
            query,
            gathered,
            ToolMessage(content="sunny and warm", tool_call_id=tool_call["id"]),
        ],
    )
    chunks = []
    first = True
    for chunk in stream:
        chunks = [*chunks, chunk]
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore[assignment]
    assert len(chunks) > 1


def test_builtin_tools_text_editor() -> None:
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")  # type: ignore[call-arg]
    tool = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
    llm_with_tools = llm.bind_tools([tool])
    response = llm_with_tools.invoke(
        "There's a syntax error in my primes.py file. Can you help me fix it?",
    )
    assert isinstance(response, AIMessage)
    assert response.tool_calls

    content_blocks = response.content_blocks
    assert len(content_blocks) == 2
    assert content_blocks[0]["type"] == "text"
    assert content_blocks[0]["text"]
    assert content_blocks[1]["type"] == "tool_call"
    assert content_blocks[1]["name"] == "str_replace_based_edit_tool"


def test_builtin_tools_computer_use() -> None:
    """Test computer use tool integration.

    Beta header should be automatically appended based on tool type.

    This test only verifies tool call generation.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
    )
    tool = {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1024,
        "display_height_px": 768,
        "display_number": 1,
    }
    llm_with_tools = llm.bind_tools([tool])
    response = llm_with_tools.invoke(
        "Can you take a screenshot to see what's on the screen?",
    )
    assert isinstance(response, AIMessage)
    assert response.tool_calls

    content_blocks = response.content_blocks
    assert len(content_blocks) >= 2
    assert content_blocks[0]["type"] == "text"
    assert content_blocks[0]["text"]

    # Check that we have a tool_call for computer use
    tool_call_blocks = [b for b in content_blocks if b["type"] == "tool_call"]
    assert len(tool_call_blocks) >= 1
    assert tool_call_blocks[0]["name"] == "computer"

    # Verify tool call has expected action (screenshot in this case)
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "computer"
    assert "action" in tool_call["args"]
    assert tool_call["args"]["action"] == "screenshot"


class GenerateUsername(BaseModel):
    """Get a username based on someone's name and hair color."""

    name: str
    hair_color: str


def test_disable_parallel_tool_calling() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    llm_with_tools = llm.bind_tools([GenerateUsername], parallel_tool_calls=False)
    result = llm_with_tools.invoke(
        "Use the GenerateUsername tool to generate user names for:\n\n"
        "Sally with green hair\n"
        "Bob with blue hair",
    )
    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) == 1


def test_anthropic_with_empty_text_block() -> None:
    """Anthropic SDK can return an empty text block."""

    @tool
    def type_letter(letter: str) -> str:
        """Type the given letter."""
        return "OK"

    model = ChatAnthropic(model=MODEL_NAME, temperature=0).bind_tools(  # type: ignore[call-arg]
        [type_letter],
    )

    messages = [
        SystemMessage(
            content="Repeat the given string using the provided tools. Do not write "
            "anything else or provide any explanations. For example, "
            "if the string is 'abc', you must print the "
            "letters 'a', 'b', and 'c' one at a time and in that order. ",
        ),
        HumanMessage(content="dog"),
        AIMessage(
            content=[
                {"text": "", "type": "text"},
                {
                    "id": "toolu_01V6d6W32QGGSmQm4BT98EKk",
                    "input": {"letter": "d"},
                    "name": "type_letter",
                    "type": "tool_use",
                },
            ],
            tool_calls=[
                {
                    "name": "type_letter",
                    "args": {"letter": "d"},
                    "id": "toolu_01V6d6W32QGGSmQm4BT98EKk",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="OK", tool_call_id="toolu_01V6d6W32QGGSmQm4BT98EKk"),
    ]

    model.invoke(messages)


def test_with_structured_output() -> None:
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )

    structured_llm = llm.with_structured_output(
        {
            "name": "get_weather",
            "description": "Get weather report for a city",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    )
    response = structured_llm.invoke("what's the weather in san francisco, ca")
    assert isinstance(response, dict)
    assert response["location"]


class Person(BaseModel):
    """Person data."""

    name: str
    age: int
    nicknames: list[str] | None


class PersonDict(TypedDict):
    """Person data as a TypedDict."""

    name: str
    age: int
    nicknames: list[str] | None


@pytest.mark.parametrize("schema", [Person, Person.model_json_schema(), PersonDict])
def test_response_format(schema: dict | type) -> None:
    model = ChatAnthropic(
        model="claude-sonnet-4-5",  # type: ignore[call-arg]
        betas=["structured-outputs-2025-11-13"],
    )
    query = "Chester (a.k.a. Chet) is 100 years old."

    response = model.invoke(query, response_format=schema)
    parsed = json.loads(response.text)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema.model_validate(parsed)
    else:
        assert isinstance(parsed, dict)
        assert parsed["name"]
        assert parsed["age"]


@pytest.mark.vcr
def test_response_format_in_agent() -> None:
    class Weather(BaseModel):
        temperature: float
        units: str

    # no tools
    agent = create_agent(
        "anthropic:claude-sonnet-4-5", response_format=ProviderStrategy(Weather)
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "75 degrees F."}]})
    assert len(result["messages"]) == 2
    parsed = json.loads(result["messages"][-1].text)
    assert Weather(**parsed) == result["structured_response"]

    # with tools
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "75 degrees Fahrenheit."

    agent = create_agent(
        "anthropic:claude-sonnet-4-5",
        tools=[get_weather],
        response_format=ProviderStrategy(Weather),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in SF?"}]},
    )
    assert len(result["messages"]) == 4
    assert result["messages"][1].tool_calls
    parsed = json.loads(result["messages"][-1].text)
    assert Weather(**parsed) == result["structured_response"]


@pytest.mark.vcr
def test_strict_tool_use() -> None:
    model = ChatAnthropic(
        model="claude-sonnet-4-5",  # type: ignore[call-arg]
        betas=["structured-outputs-2025-11-13"],
    )

    def get_weather(location: str, unit: Literal["C", "F"]) -> str:
        """Get the weather at a location."""
        return "75 degrees Fahrenheit."

    model_with_tools = model.bind_tools([get_weather], strict=True)

    response = model_with_tools.invoke("What's the weather in Boston, in Celsius?")
    assert response.tool_calls


def test_get_num_tokens_from_messages() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]

    # Test simple case
    messages = [
        SystemMessage(content="You are a scientist"),
        HumanMessage(content="Hello, Claude"),
    ]
    num_tokens = llm.get_num_tokens_from_messages(messages)
    assert num_tokens > 0

    # Test tool use
    @tool(parse_docstring=True)
    def get_weather(location: str) -> str:
        """Get the current weather in a given location.

        Args:
            location: The city and state, e.g. San Francisco, CA

        """
        return "Sunny"

    messages = [
        HumanMessage(content="What's the weather like in San Francisco?"),
    ]
    num_tokens = llm.get_num_tokens_from_messages(messages, tools=[get_weather])
    assert num_tokens > 0

    messages = [
        HumanMessage(content="What's the weather like in San Francisco?"),
        AIMessage(
            content=[
                {"text": "Let's see.", "type": "text"},
                {
                    "id": "toolu_01V6d6W32QGGSmQm4BT98EKk",
                    "input": {"location": "SF"},
                    "name": "get_weather",
                    "type": "tool_use",
                },
            ],
            tool_calls=[
                {
                    "name": "get_weather",
                    "args": {"location": "SF"},
                    "id": "toolu_01V6d6W32QGGSmQm4BT98EKk",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content="Sunny", tool_call_id="toolu_01V6d6W32QGGSmQm4BT98EKk"),
    ]
    num_tokens = llm.get_num_tokens_from_messages(messages, tools=[get_weather])
    assert num_tokens > 0


class GetWeather(BaseModel):
    """Get the current weather in a given location."""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


@pytest.mark.parametrize("tool_choice", ["GetWeather", "auto", "any"])
def test_anthropic_bind_tools_tool_choice(tool_choice: str) -> None:
    chat_model = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice=tool_choice)
    response = chat_model_with_tools.invoke("what's the weather in ny and la")
    assert isinstance(response, AIMessage)


def test_pdf_document_input() -> None:
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    data = b64encode(requests.get(url, timeout=10).content).decode()

    result = ChatAnthropic(model=MODEL_NAME).invoke(  # type: ignore[call-arg]
        [
            HumanMessage(
                [
                    "summarize this document",
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "data": data,
                            "media_type": "application/pdf",
                        },
                    },
                ],
            ),
        ],
    )
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, str)
    assert len(result.content) > 0


@pytest.mark.default_cassette("test_agent_loop.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_agent_loop(output_version: Literal["v0", "v1"]) -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatAnthropic(model=MODEL_NAME, output_version=output_version)  # type: ignore[call-arg]
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)


@pytest.mark.default_cassette("test_agent_loop_streaming.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_agent_loop_streaming(output_version: Literal["v0", "v1"]) -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatAnthropic(
        model=MODEL_NAME,
        streaming=True,
        output_version=output_version,  # type: ignore[call-arg]
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)

    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)


@pytest.mark.default_cassette("test_citations.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_citations(output_version: Literal["v0", "v1"]) -> None:
    llm = ChatAnthropic(model=MODEL_NAME, output_version=output_version)  # type: ignore[call-arg]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "content",
                        "content": [
                            {"type": "text", "text": "The grass is green"},
                            {"type": "text", "text": "The sky is blue"},
                        ],
                    },
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": "What color is the grass and sky?"},
            ],
        },
    ]
    response = llm.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    if output_version == "v1":
        assert any("annotations" in block for block in response.content)
    else:
        assert any("citations" in block for block in response.content)

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream(messages):
        full = cast("BaseMessageChunk", chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert not any("citation" in block for block in full.content)
    if output_version == "v1":
        assert any("annotations" in block for block in full.content)
    else:
        assert any("citations" in block for block in full.content)

    # Test pass back in
    next_message = {
        "role": "user",
        "content": "Can you comment on the citations you just made?",
    }
    _ = llm.invoke([*messages, full, next_message])


@pytest.mark.vcr
def test_thinking() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
    )

    input_message = {"role": "user", "content": "Hello"}
    response = llm.invoke([input_message])
    assert any("thinking" in block for block in response.content)
    for block in response.content:
        assert isinstance(block, dict)
        if block["type"] == "thinking":
            assert set(block.keys()) == {"type", "thinking", "signature"}
            assert block["thinking"]
            assert isinstance(block["thinking"], str)
            assert block["signature"]
            assert isinstance(block["signature"], str)

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = cast("BaseMessageChunk", chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert any("thinking" in block for block in full.content)
    for block in full.content:
        assert isinstance(block, dict)
        if block["type"] == "thinking":
            assert set(block.keys()) == {"type", "thinking", "signature", "index"}
            assert block["thinking"]
            assert isinstance(block["thinking"], str)
            assert block["signature"]
            assert isinstance(block["signature"], str)

    # Test pass back in
    next_message = {"role": "user", "content": "How are you?"}
    _ = llm.invoke([input_message, full, next_message])


@pytest.mark.default_cassette("test_thinking.yaml.gz")
@pytest.mark.vcr
def test_thinking_v1() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
        output_version="v1",
    )

    input_message = {"role": "user", "content": "Hello"}
    response = llm.invoke([input_message])
    assert any("reasoning" in block for block in response.content)
    for block in response.content:
        assert isinstance(block, dict)
        if block["type"] == "reasoning":
            assert set(block.keys()) == {"type", "reasoning", "extras"}
            assert block["reasoning"]
            assert isinstance(block["reasoning"], str)
            signature = block["extras"]["signature"]
            assert signature
            assert isinstance(signature, str)

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = cast(BaseMessageChunk, chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert any("reasoning" in block for block in full.content)
    for block in full.content:
        assert isinstance(block, dict)
        if block["type"] == "reasoning":
            assert set(block.keys()) == {"type", "reasoning", "extras", "index"}
            assert block["reasoning"]
            assert isinstance(block["reasoning"], str)
            signature = block["extras"]["signature"]
            assert signature
            assert isinstance(signature, str)

    # Test pass back in
    next_message = {"role": "user", "content": "How are you?"}
    _ = llm.invoke([input_message, full, next_message])


@pytest.mark.default_cassette("test_redacted_thinking.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_redacted_thinking(output_version: Literal["v0", "v1"]) -> None:
    llm = ChatAnthropic(
        # It appears that Sonnet 4.5 either: isn't returning redacted thinking blocks,
        # or the magic string is broken? Retry later once 3-7 finally removed
        model="claude-3-7-sonnet-latest",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
        output_version=output_version,
    )
    query = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"  # noqa: E501
    input_message = {"role": "user", "content": query}

    response = llm.invoke([input_message])
    value = None
    for block in response.content:
        assert isinstance(block, dict)
        if block["type"] == "redacted_thinking":
            value = block
        elif (
            block["type"] == "non_standard"
            and block["value"]["type"] == "redacted_thinking"
        ):
            value = block["value"]
        else:
            pass
        if value:
            assert set(value.keys()) == {"type", "data"}
            assert value["data"]
            assert isinstance(value["data"], str)
    assert value is not None

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        full = cast("BaseMessageChunk", chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    value = None
    for block in full.content:
        assert isinstance(block, dict)
        if block["type"] == "redacted_thinking":
            value = block
            assert set(value.keys()) == {"type", "data", "index"}
            assert "index" in block
        elif (
            block["type"] == "non_standard"
            and block["value"]["type"] == "redacted_thinking"
        ):
            value = block["value"]
            assert isinstance(value, dict)
            assert set(value.keys()) == {"type", "data"}
            assert "index" in block
        else:
            pass
        if value:
            assert value["data"]
            assert isinstance(value["data"], str)
    assert value is not None

    # Test pass back in
    next_message = {"role": "user", "content": "What?"}
    _ = llm.invoke([input_message, full, next_message])


def test_structured_output_thinking_enabled() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
    )
    with pytest.warns(match="structured output"):
        structured_llm = llm.with_structured_output(GenerateUsername)
    query = "Generate a username for Sally with green hair"
    response = structured_llm.invoke(query)
    assert isinstance(response, GenerateUsername)

    with pytest.raises(OutputParserException):
        structured_llm.invoke("Hello")

    # Test streaming
    for chunk in structured_llm.stream(query):
        assert isinstance(chunk, GenerateUsername)


def test_structured_output_thinking_force_tool_use() -> None:
    # Structured output currently relies on forced tool use, which is not supported
    # when `thinking` is enabled. When this test fails, it means that the feature
    # is supported and the workarounds in `with_structured_output` should be removed.
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
    ).bind_tools(
        [GenerateUsername],
        tool_choice="GenerateUsername",
    )
    with pytest.raises(BadRequestError):
        llm.invoke("Generate a username for Sally with green hair")


def test_effort_parameter() -> None:
    """Test that effort parameter can be passed without errors.

    Only Opus 4.5 supports currently.
    """
    llm = ChatAnthropic(
        model="claude-opus-4-5-20251101",
        effort="medium",
        max_tokens=100,
    )

    result = llm.invoke("Say hello in one sentence")

    # Verify we got a response
    assert isinstance(result.content, str)
    assert len(result.content) > 0

    # Verify response metadata is present
    assert "model_name" in result.response_metadata
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] > 0
    assert result.usage_metadata["output_tokens"] > 0


def test_image_tool_calling() -> None:
    """Test tool calling with image inputs."""

    class color_picker(BaseModel):  # noqa: N801
        """Input your fav color and get a random fact about it."""

        fav_color: str

    human_content: list[dict] = [
        {
            "type": "text",
            "text": "what's your favorite color in this image",
        },
    ]
    image_url = "https://raw.githubusercontent.com/langchain-ai/docs/4d11d08b6b0e210bd456943f7a22febbd168b543/src/images/agentic-rag-output.png"
    image_data = b64encode(httpx.get(image_url).content).decode("utf-8")
    human_content.append(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data,
            },
        },
    )
    messages = [
        SystemMessage("you're a good assistant"),
        HumanMessage(human_content),  # type: ignore[arg-type]
        AIMessage(
            [
                {"type": "text", "text": "Hmm let me think about that"},
                {
                    "type": "tool_use",
                    "input": {"fav_color": "purple"},
                    "id": "foo",
                    "name": "color_picker",
                },
            ],
        ),
        HumanMessage(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "foo",
                    "content": [
                        {
                            "type": "text",
                            "text": "purple is a great pick! that's my sister's favorite color",  # noqa: E501
                        },
                    ],
                    "is_error": False,
                },
                {"type": "text", "text": "what's my sister's favorite color"},
            ],
        ),
    ]
    llm = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    _ = llm.bind_tools([color_picker]).invoke(messages)


@pytest.mark.default_cassette("test_web_search.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_web_search(output_version: Literal["v0", "v1"]) -> None:
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        max_tokens=1024,
        output_version=output_version,
    )

    tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "How do I update a web app to TypeScript 5.5?",
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "server_tool_use", "web_search_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "server_tool_use", "web_search_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please repeat the last search, but focus on sources from 2024.",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )


@pytest.mark.vcr
def test_web_fetch() -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once it's generally available.
    """
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        max_tokens=1024,
        betas=["web-fetch-2025-09-10"],
    )
    tool = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1}
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Fetch the content at https://docs.langchain.com and analyze",
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {
        block["type"] for block in response.content if isinstance(block, dict)
    }

    # A successful fetch call should include:
    # 1. text response from the model (e.g. "I'll fetch that for you")
    # 2. server_tool_use block indicating the tool was called (using tool "web_fetch")
    # 3. web_fetch_tool_result block with the results of said fetch
    assert block_types == {"text", "server_tool_use", "web_fetch_tool_result"}

    # Verify web fetch result structure
    web_fetch_results = [
        block
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    assert len(web_fetch_results) == 1  # Since max_uses=1
    fetch_result = web_fetch_results[0]
    assert "content" in fetch_result
    assert "url" in fetch_result["content"]
    assert "retrieved_at" in fetch_result["content"]

    # Fetch with citations enabled
    tool_with_citations = tool.copy()
    tool_with_citations["citations"] = {"enabled": True}
    llm_with_citations = llm.bind_tools([tool_with_citations])

    citation_message = {
        "role": "user",
        "content": (
            "Fetch https://docs.langchain.com and provide specific quotes with "
            "citations"
        ),
    }
    citation_response = llm_with_citations.invoke([citation_message])

    citation_results = [
        block
        for block in citation_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    assert len(citation_results) == 1  # Since max_uses=1
    citation_result = citation_results[0]
    assert citation_result["content"]["content"]["citations"]["enabled"]
    text_blocks = [
        block
        for block in citation_response.content
        if isinstance(block, dict) and block.get("type") == "text"
    ]

    # Check that the response contains actual citations in the content
    has_citations = False
    for block in text_blocks:
        citations = block.get("citations", [])
        for citation in citations:
            if citation.get("type") and citation.get("start_char_index"):
                has_citations = True
                break
    assert has_citations, (
        "Expected inline citation tags in response when citations are enabled for "
        "web fetch"
    )

    # Max content tokens param
    tool_with_limit = tool.copy()
    tool_with_limit["max_content_tokens"] = 1000
    llm_with_limit = llm.bind_tools([tool_with_limit])

    limit_response = llm_with_limit.invoke([input_message])
    # Response should still work even with content limits
    assert any(
        block["type"] == "web_fetch_tool_result"
        for block in limit_response.content
        if isinstance(block, dict)
    )

    # Domains filtering (note: only one can be set at a time)
    tool_with_allowed_domains = tool.copy()
    tool_with_allowed_domains["allowed_domains"] = ["docs.langchain.com"]
    llm_with_allowed = llm.bind_tools([tool_with_allowed_domains])

    allowed_response = llm_with_allowed.invoke([input_message])
    assert any(
        block["type"] == "web_fetch_tool_result"
        for block in allowed_response.content
        if isinstance(block, dict)
    )

    # Test that a disallowed domain doesn't work
    tool_with_disallowed_domains = tool.copy()
    tool_with_disallowed_domains["allowed_domains"] = [
        "example.com"
    ]  # Not docs.langchain.com
    llm_with_disallowed = llm.bind_tools([tool_with_disallowed_domains])

    disallowed_response = llm_with_disallowed.invoke([input_message])

    # We should get an error result since the domain (docs.langchain.com) is not allowed
    disallowed_results = [
        block
        for block in disallowed_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    if disallowed_results:
        disallowed_result = disallowed_results[0]
        if disallowed_result.get("content", {}).get("type") == "web_fetch_tool_error":
            assert disallowed_result["content"]["error_code"] in [
                "invalid_url",
                "fetch_failed",
            ]

    # Blocked domains filtering
    tool_with_blocked_domains = tool.copy()
    tool_with_blocked_domains["blocked_domains"] = ["example.com"]
    llm_with_blocked = llm.bind_tools([tool_with_blocked_domains])

    blocked_response = llm_with_blocked.invoke([input_message])
    assert any(
        block["type"] == "web_fetch_tool_result"
        for block in blocked_response.content
        if isinstance(block, dict)
    )

    # Test fetching from a blocked domain fails
    blocked_domain_message = {
        "role": "user",
        "content": "Fetch https://example.com and analyze",
    }
    tool_with_blocked_example = tool.copy()
    tool_with_blocked_example["blocked_domains"] = ["example.com"]
    llm_with_blocked_example = llm.bind_tools([tool_with_blocked_example])

    blocked_domain_response = llm_with_blocked_example.invoke([blocked_domain_message])

    # Should get an error when trying to access a blocked domain
    blocked_domain_results = [
        block
        for block in blocked_domain_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    if blocked_domain_results:
        blocked_result = blocked_domain_results[0]
        if blocked_result.get("content", {}).get("type") == "web_fetch_tool_error":
            assert blocked_result["content"]["error_code"] in [
                "invalid_url",
                "fetch_failed",
            ]

    # Max uses parameter - test exceeding the limit
    multi_fetch_message = {
        "role": "user",
        "content": (
            "Fetch https://docs.langchain.com and then try to fetch "
            "https://langchain.com"
        ),
    }
    max_uses_response = llm_with_tools.invoke([multi_fetch_message])

    # Should contain at least one fetch result and potentially an error for the second
    fetch_results = [
        block
        for block in max_uses_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]  # type: ignore[index]
    assert len(fetch_results) >= 1
    error_results = [
        r
        for r in fetch_results
        if r.get("content", {}).get("type") == "web_fetch_tool_error"
    ]
    if error_results:
        assert any(
            r["content"]["error_code"] == "max_uses_exceeded" for r in error_results
        )

    # Streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content if isinstance(block, dict)}
    assert block_types == {"text", "server_tool_use", "web_fetch_tool_result"}

    # Test that URLs from context can be used in follow-up
    next_message = {
        "role": "user",
        "content": "What does the site you just fetched say about models?",
    }
    follow_up_response = llm_with_tools.invoke(
        [input_message, full, next_message],
    )
    # Should work without issues since URL was already in context
    assert isinstance(follow_up_response.content, (list, str))

    # Error handling - test with an invalid URL format
    error_message = {
        "role": "user",
        "content": "Try to fetch this invalid URL: not-a-valid-url",
    }
    error_response = llm_with_tools.invoke([error_message])

    # Should handle the error gracefully
    assert isinstance(error_response.content, (list, str))

    # PDF document fetching
    pdf_message = {
        "role": "user",
        "content": (
            "Fetch this PDF: "
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf "
            "and summarize its content",
        ),
    }
    pdf_response = llm_with_tools.invoke([pdf_message])

    assert any(
        block["type"] == "web_fetch_tool_result"
        for block in pdf_response.content
        if isinstance(block, dict)
    )

    # Verify PDF content structure (should have base64 data for PDFs)
    pdf_results = [
        block
        for block in pdf_response.content
        if isinstance(block, dict) and block.get("type") == "web_fetch_tool_result"
    ]
    if pdf_results:
        pdf_result = pdf_results[0]
        content = pdf_result.get("content", {})
        if content.get("content", {}).get("source", {}).get("type") == "base64":
            assert content["content"]["source"]["media_type"] == "application/pdf"
            assert "data" in content["content"]["source"]


@pytest.mark.default_cassette("test_web_fetch_v1.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_web_fetch_v1(output_version: Literal["v0", "v1"]) -> None:
    """Test that http calls are unchanged between v0 and v1."""
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["web-fetch-2025-09-10"],
        output_version=output_version,
    )

    if output_version == "v0":
        call_key = "server_tool_use"
        result_key = "web_fetch_tool_result"
    else:
        # v1
        call_key = "server_tool_call"
        result_key = "server_tool_result"

    tool = {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 1,
        "citations": {"enabled": True},
    }
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Fetch the content at https://docs.langchain.com and analyze",
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    assert block_types == {"text", call_key, result_key}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    assert block_types == {"text", call_key, result_key}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "What does the site you just fetched say about models?",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )


@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_code_execution_old(output_version: Literal["v0", "v1"]) -> None:
    """Note: this tests the `code_execution_20250522` tool, which is now legacy.

    See the `test_code_execution` test below to test the current
    `code_execution_20250825` tool.

    Migration guide: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#upgrade-to-latest-tool-version
    """
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["code-execution-2025-05-22"],
        output_version=output_version,
    )

    tool = {"type": "code_execution_20250522", "name": "code_execution"}
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Calculate the mean and standard deviation of "
                    "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
                ),
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "server_tool_use", "code_execution_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "server_tool_use", "code_execution_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please add more comments to the code.",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )


@pytest.mark.default_cassette("test_code_execution.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_code_execution(output_version: Literal["v0", "v1"]) -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once generally available.
    """
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["code-execution-2025-08-25"],
        output_version=output_version,
    )

    tool = {"type": "code_execution_20250825", "name": "code_execution"}
    llm_with_tools = llm.bind_tools([tool])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Calculate the mean and standard deviation of "
                    "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
                ),
            },
        ],
    }
    response = llm_with_tools.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {
            "text",
            "server_tool_use",
            "text_editor_code_execution_tool_result",
            "bash_code_execution_tool_result",
        }
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {
            "text",
            "server_tool_use",
            "text_editor_code_execution_tool_result",
            "bash_code_execution_tool_result",
        }
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please add more comments to the code.",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )


@pytest.mark.default_cassette("test_remote_mcp.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_remote_mcp(output_version: Literal["v0", "v1"]) -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once generally available.
    """
    mcp_servers = [
        {
            "type": "url",
            "url": "https://mcp.deepwiki.com/mcp",
            "name": "deepwiki",
            "authorization_token": "PLACEHOLDER",
        },
    ]

    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        mcp_servers=mcp_servers,
        output_version=output_version,
    ).bind_tools([{"type": "mcp_toolset", "mcp_server_name": "deepwiki"}])

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "What transport protocols does the 2025-03-26 version of the MCP "
                    "spec (modelcontextprotocol/modelcontextprotocol) support?"
                ),
            },
        ],
    }
    response = llm.invoke([input_message])
    assert all(isinstance(block, dict) for block in response.content)
    block_types = {block["type"] for block in response.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "mcp_tool_use", "mcp_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert all(isinstance(block, dict) for block in full.content)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "mcp_tool_use", "mcp_tool_result"}
    else:
        assert block_types == {"text", "server_tool_call", "server_tool_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please query the same tool again, but add 'please' to your query.",
    }
    _ = llm.invoke(
        [input_message, full, next_message],
    )


@pytest.mark.parametrize("block_format", ["anthropic", "standard"])
def test_files_api_image(block_format: str) -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once generally available.
    """
    image_file_id = os.getenv("ANTHROPIC_FILES_API_IMAGE_ID")
    if not image_file_id:
        pytest.skip()
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["files-api-2025-04-14"],
    )
    if block_format == "anthropic":
        block = {
            "type": "image",
            "source": {
                "type": "file",
                "file_id": image_file_id,
            },
        }
    else:
        # standard block format
        block = {
            "type": "image",
            "file_id": image_file_id,
        }
    input_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            block,
        ],
    }
    _ = llm.invoke([input_message])


@pytest.mark.parametrize("block_format", ["anthropic", "standard"])
def test_files_api_pdf(block_format: str) -> None:
    """Note: this is a beta feature.

    TODO: Update to remove beta once generally available.
    """
    pdf_file_id = os.getenv("ANTHROPIC_FILES_API_PDF_ID")
    if not pdf_file_id:
        pytest.skip()
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        betas=["files-api-2025-04-14"],
    )
    if block_format == "anthropic":
        block = {"type": "document", "source": {"type": "file", "file_id": pdf_file_id}}
    else:
        # standard block format
        block = {
            "type": "file",
            "file_id": pdf_file_id,
        }
    input_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this document."},
            block,
        ],
    }
    _ = llm.invoke([input_message])


@pytest.mark.vcr
def test_search_result_tool_message() -> None:
    """Test that we can pass a search result tool message to the model."""
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )

    @tool
    def retrieval_tool(query: str) -> list[dict]:
        """Retrieve information from a knowledge base."""
        return [
            {
                "type": "search_result",
                "title": "Leave policy",
                "source": "HR Leave Policy 2025",
                "citations": {"enabled": True},
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "To request vacation days, submit a leave request form "
                            "through the HR portal. Approval will be sent by email."
                        ),
                    },
                ],
            },
        ]

    tool_call = {
        "type": "tool_call",
        "name": "retrieval_tool",
        "args": {"query": "vacation days request process"},
        "id": "toolu_abc123",
    }

    tool_message = retrieval_tool.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    assert isinstance(tool_message.content, list)

    messages = [
        HumanMessage("How do I request vacation days?"),
        AIMessage(
            [{"type": "text", "text": "Let me look that up for you."}],
            tool_calls=[tool_call],
        ),
        tool_message,
    ]

    result = llm.invoke(messages)
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)
    assert any("citations" in block for block in result.content)

    assert (
        _convert_from_v1_to_anthropic(result.content_blocks, [], "anthropic")
        == result.content
    )


def test_search_result_top_level() -> None:
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
    )
    input_message = HumanMessage(
        [
            {
                "type": "search_result",
                "title": "Leave policy",
                "source": "HR Leave Policy 2025 - page 1",
                "citations": {"enabled": True},
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "To request vacation days, submit a leave request form "
                            "through the HR portal. Approval will be sent by email."
                        ),
                    },
                ],
            },
            {
                "type": "search_result",
                "title": "Leave policy",
                "source": "HR Leave Policy 2025 - page 2",
                "citations": {"enabled": True},
                "content": [
                    {
                        "type": "text",
                        "text": "Managers have 3 days to approve a request.",
                    },
                ],
            },
            {
                "type": "text",
                "text": "How do I request vacation days?",
            },
        ],
    )
    result = llm.invoke([input_message])
    assert isinstance(result, AIMessage)
    assert isinstance(result.content, list)
    assert any("citations" in block for block in result.content)

    assert (
        _convert_from_v1_to_anthropic(result.content_blocks, [], "anthropic")
        == result.content
    )


def test_memory_tool() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        betas=["context-management-2025-06-27"],
    )
    llm_with_tools = llm.bind_tools([{"type": "memory_20250818", "name": "memory"}])
    response = llm_with_tools.invoke("What are my interests?")
    assert isinstance(response, AIMessage)
    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "memory"


@pytest.mark.vcr
def test_context_management() -> None:
    # TODO: update example to trigger action
    llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",  # type: ignore[call-arg]
        betas=["context-management-2025-06-27"],
        context_management={
            "edits": [
                {
                    "type": "clear_tool_uses_20250919",
                    "trigger": {"type": "input_tokens", "value": 10},
                    "clear_at_least": {"type": "input_tokens", "value": 5},
                }
            ]
        },
        max_tokens=1024,  # type: ignore[call-arg]
    )
    llm_with_tools = llm.bind_tools(
        [{"type": "web_search_20250305", "name": "web_search"}]
    )
    input_message = {"role": "user", "content": "Search for recent developments in AI"}
    response = llm_with_tools.invoke([input_message])
    assert response.response_metadata.get("context_management")

    # Test streaming
    full: BaseMessageChunk | None = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata.get("context_management")


@pytest.mark.default_cassette("test_tool_search.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_tool_search(output_version: str) -> None:
    """Test tool search with LangChain tools using extras parameter."""

    @tool(parse_docstring=True, extras={"defer_loading": True})
    def get_weather(location: str, unit: str = "fahrenheit") -> str:
        """Get the current weather for a location.

        Args:
            location: City name
            unit: Temperature unit (celsius or fahrenheit)
        """
        return f"The weather in {location} is sunny and 72°{unit[0].upper()}"

    @tool(parse_docstring=True, extras={"defer_loading": True})
    def search_files(query: str) -> str:
        """Search through files in the workspace.

        Args:
            query: Search query
        """
        return f"Found 3 files matching '{query}'"

    model = ChatAnthropic(
        model="claude-opus-4-5-20251101", output_version=output_version
    )

    agent = create_agent(  # type: ignore[var-annotated]
        model,
        tools=[
            {
                "type": "tool_search_tool_regex_20251119",
                "name": "tool_search_tool_regex",
            },
            get_weather,
            search_files,
        ],
    )

    # Test with actual API call
    input_message = {
        "role": "user",
        "content": "What's the weather in San Francisco? Find and use a tool.",
    }
    result = agent.invoke({"messages": [input_message]})
    first_response = result["messages"][1]
    content_types = [block["type"] for block in first_response.content]
    if output_version == "v0":
        assert content_types == [
            "text",
            "server_tool_use",
            "tool_search_tool_result",
            "text",
            "tool_use",
        ]
    else:
        # v1
        assert content_types == [
            "text",
            "server_tool_call",
            "server_tool_result",
            "text",
            "tool_call",
        ]

    answer = result["messages"][-1]
    assert not answer.tool_calls
    assert answer.text


@pytest.mark.default_cassette("test_programmatic_tool_use.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_programmatic_tool_use(output_version: str) -> None:
    """Test programmatic tool use.

    Implicitly checks that `allowed_callers` in tool extras works.
    """

    @tool(extras={"allowed_callers": ["code_execution_20250825"]})
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "It's sunny."

    tools: list = [
        {"type": "code_execution_20250825", "name": "code_execution"},
        get_weather,
    ]

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        betas=["advanced-tool-use-2025-11-20"],
        reuse_last_container=True,
        output_version=output_version,
    )

    agent = create_agent(model, tools=tools)  # type: ignore[var-annotated]

    input_query = {
        "role": "user",
        "content": "What's the weather in Boston?",
    }

    result = agent.invoke({"messages": [input_query]})
    assert len(result["messages"]) == 4
    tool_call_message = result["messages"][1]
    response_message = result["messages"][-1]

    if output_version == "v0":
        server_tool_use_block = next(
            block
            for block in tool_call_message.content
            if block["type"] == "server_tool_use"
        )
        assert server_tool_use_block

        tool_use_block = next(
            block for block in tool_call_message.content if block["type"] == "tool_use"
        )
        assert "caller" in tool_use_block

        code_execution_result = next(
            block
            for block in response_message.content
            if block["type"] == "code_execution_tool_result"
        )
        assert code_execution_result["content"]["return_code"] == 0
    else:
        server_tool_call_block = next(
            block
            for block in tool_call_message.content
            if block["type"] == "server_tool_call"
        )
        assert server_tool_call_block

        tool_call_block = next(
            block for block in tool_call_message.content if block["type"] == "tool_call"
        )
        assert "caller" in tool_call_block["extras"]

        server_tool_result = next(
            block
            for block in response_message.content
            if block["type"] == "server_tool_result"
        )
        assert server_tool_result["output"]["return_code"] == 0


@pytest.mark.default_cassette("test_programmatic_tool_use_streaming.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_programmatic_tool_use_streaming(output_version: str) -> None:
    @tool(extras={"allowed_callers": ["code_execution_20250825"]})
    def get_weather(location: str) -> str:
        """Get the weather at a location."""
        return "It's sunny."

    tools: list = [
        {"type": "code_execution_20250825", "name": "code_execution"},
        get_weather,
    ]

    model = ChatAnthropic(
        model="claude-sonnet-4-5",
        betas=["advanced-tool-use-2025-11-20"],
        reuse_last_container=True,
        streaming=True,
        output_version=output_version,
    )

    agent = create_agent(model, tools=tools)  # type: ignore[var-annotated]

    input_query = {
        "role": "user",
        "content": "What's the weather in Boston?",
    }

    result = agent.invoke({"messages": [input_query]})
    assert len(result["messages"]) == 4
    tool_call_message = result["messages"][1]
    response_message = result["messages"][-1]

    if output_version == "v0":
        server_tool_use_block = next(
            block
            for block in tool_call_message.content
            if block["type"] == "server_tool_use"
        )
        assert server_tool_use_block

        tool_use_block = next(
            block for block in tool_call_message.content if block["type"] == "tool_use"
        )
        assert "caller" in tool_use_block

        code_execution_result = next(
            block
            for block in response_message.content
            if block["type"] == "code_execution_tool_result"
        )
        assert code_execution_result["content"]["return_code"] == 0
    else:
        server_tool_call_block = next(
            block
            for block in tool_call_message.content
            if block["type"] == "server_tool_call"
        )
        assert server_tool_call_block

        tool_call_block = next(
            block for block in tool_call_message.content if block["type"] == "tool_call"
        )
        assert "caller" in tool_call_block["extras"]

        server_tool_result = next(
            block
            for block in response_message.content
            if block["type"] == "server_tool_result"
        )
        assert server_tool_result["output"]["return_code"] == 0


def test_async_shared_client() -> None:
    llm = ChatAnthropic(model=MODEL_NAME)  # type: ignore[call-arg]
    _ = asyncio.run(llm.ainvoke("Hello"))
    _ = asyncio.run(llm.ainvoke("Hello"))


def test_fine_grained_tool_streaming() -> None:
    """Test fine-grained tool streaming reduces latency for tool parameter streaming.

    Fine-grained tool streaming enables Claude to stream tool parameter values.

    https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming
    """
    llm = ChatAnthropic(
        model=MODEL_NAME,  # type: ignore[call-arg]
        temperature=0,
        betas=["fine-grained-tool-streaming-2025-05-14"],
    )

    # Define a tool that requires a longer text parameter
    tool_definition = {
        "name": "write_document",
        "description": "Write a document with the given content",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title"},
                "content": {
                    "type": "string",
                    "description": "The full document content",
                },
            },
            "required": ["title", "content"],
        },
    }

    llm_with_tools = llm.bind_tools([tool_definition])
    query = (
        "Write a document about the benefits of streaming APIs. "
        "Include at least 3 paragraphs."
    )

    # Test streaming with fine-grained tool streaming
    first = True
    chunks: list[BaseMessage | BaseMessageChunk] = []
    tool_call_chunks = []

    for chunk in llm_with_tools.stream(query):
        chunks.append(chunk)
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore[assignment]

        # Collect tool call chunks
        tool_call_chunks.extend(
            [
                block
                for block in chunk.content_blocks
                if block["type"] == "tool_call_chunk"
            ]
        )

    # Verify we got chunks
    assert len(chunks) > 1

    # Verify final message has tool call
    assert isinstance(gathered, AIMessageChunk)
    assert isinstance(gathered.tool_calls, list)
    assert len(gathered.tool_calls) >= 1

    # Find the write_document tool call
    write_doc_call = None
    for tool_call in gathered.tool_calls:
        if tool_call["name"] == "write_document":
            write_doc_call = tool_call
            break

    assert write_doc_call is not None, "write_document tool call not found"
    assert isinstance(write_doc_call["args"], dict)
    assert "title" in write_doc_call["args"]
    assert "content" in write_doc_call["args"]
    assert (
        len(write_doc_call["args"]["content"]) > 100
    )  # Should have substantial content

    # Verify tool_call_chunks were received
    # With fine-grained streaming, we should get tool call chunks
    assert len(tool_call_chunks) > 0

    # Verify content_blocks in final message
    content_blocks = gathered.content_blocks
    assert len(content_blocks) >= 1

    # Should have at least one tool_call block
    tool_call_blocks = [b for b in content_blocks if b["type"] == "tool_call"]
    assert len(tool_call_blocks) >= 1

    write_doc_block = None
    for block in tool_call_blocks:
        if block["name"] == "write_document":
            write_doc_block = block
            break

    assert write_doc_block is not None
    assert write_doc_block["name"] == "write_document"
    assert "args" in write_doc_block

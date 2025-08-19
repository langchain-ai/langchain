"""Test ChatAnthropic chat model."""

from __future__ import annotations

import asyncio
import json
import os
from base64 import b64encode
from typing import Literal, Optional, cast

import httpx
import pytest
import requests
from anthropic import BadRequestError
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

from langchain_anthropic import ChatAnthropic, ChatAnthropicMessages
from tests.unit_tests._utils import FakeCallbackHandler

MODEL_NAME = "claude-3-5-haiku-latest"
IMAGE_MODEL_NAME = "claude-3-5-haiku-latest"


def test_stream() -> None:
    """Test streaming tokens from Anthropic."""
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    full: Optional[BaseMessageChunk] = None
    chunks_with_input_token_counts = 0
    chunks_with_output_token_counts = 0
    chunks_with_model_name = 0
    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)
        full = cast(BaseMessageChunk, token) if full is None else full + token
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
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    full: Optional[BaseMessageChunk] = None
    chunks_with_input_token_counts = 0
    chunks_with_output_token_counts = 0
    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)
        full = cast(BaseMessageChunk, token) if full is None else full + token
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
    """Test streaming tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

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
    """Test batch tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)
    assert "model_name" in result.response_metadata


def test_invoke() -> None:
    """Test invoke tokens from ChatAnthropicMessages."""
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

    result = llm.invoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_system_invoke() -> None:
    """Test invoke tokens with a system message."""
    llm = ChatAnthropicMessages(model_name=MODEL_NAME)  # type: ignore[call-arg, call-arg]

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
        callback_manager=callback_manager,
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
        callback_manager=callback_manager,
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
    chat = ChatAnthropic(model=IMAGE_MODEL_NAME)  # type: ignore[call-arg]
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

    llm = ChatAnthropicMessages(  # type: ignore[call-arg, call-arg]
        model_name=MODEL_NAME,
        streaming=True,
        callback_manager=callback_manager,
    )

    response = llm.generate([[HumanMessage(content="I'm Pickle Rick")]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)


async def test_astreaming() -> None:
    """Test streaming tokens from Anthropic."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])

    llm = ChatAnthropicMessages(  # type: ignore[call-arg, call-arg]
        model_name=MODEL_NAME,
        streaming=True,
        callback_manager=callback_manager,
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
    assert content_blocks[1]["type"] == "tool_call_chunk"
    assert content_blocks[1]["name"] == "get_weather"
    assert content_blocks[1]["args"]

    # Testing token-efficient tools
    # https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use
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


def test_builtin_tools() -> None:
    llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")  # type: ignore[call-arg]
    tool = {"type": "text_editor_20250124", "name": "str_replace_editor"}
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
    assert content_blocks[1]["name"] == "str_replace_editor"


class GenerateUsername(BaseModel):
    """Get a username based on someone's name and hair color."""

    name: str
    hair_color: str


def test_disable_parallel_tool_calling() -> None:
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")  # type: ignore[call-arg]
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

    model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0).bind_tools(  # type: ignore[call-arg]
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
        model="claude-3-opus-20240229",  # type: ignore[call-arg]
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


def test_get_num_tokens_from_messages() -> None:
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")  # type: ignore[call-arg]

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

    result = ChatAnthropic(model=IMAGE_MODEL_NAME).invoke(  # type: ignore[call-arg]
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

    llm = ChatAnthropic(model="claude-3-5-haiku-latest", output_version=output_version)  # type: ignore[call-arg]
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
        model="claude-3-5-haiku-latest", streaming=True, output_version=output_version  # type: ignore[call-arg]
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
    llm = ChatAnthropic(model="claude-3-5-haiku-latest", output_version=output_version)  # type: ignore[call-arg]
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
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream(messages):
        full = cast(BaseMessageChunk, chunk) if full is None else full + chunk
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
        model="claude-3-7-sonnet-latest",  # type: ignore[call-arg]
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
            assert block["thinking"] and isinstance(block["thinking"], str)
            assert block["signature"] and isinstance(block["signature"], str)

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message]):
        full = cast(BaseMessageChunk, chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert any("thinking" in block for block in full.content)
    for block in full.content:
        assert isinstance(block, dict)
        if block["type"] == "thinking":
            assert set(block.keys()) == {"type", "thinking", "signature", "index"}
            assert block["thinking"] and isinstance(block["thinking"], str)
            assert block["signature"] and isinstance(block["signature"], str)

    # Test pass back in
    next_message = {"role": "user", "content": "How are you?"}
    _ = llm.invoke([input_message, full, next_message])


@pytest.mark.default_cassette("test_thinking.yaml.gz")
@pytest.mark.vcr
def test_thinking_v1() -> None:
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-latest",  # type: ignore[call-arg]
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
            assert block["reasoning"] and isinstance(block["reasoning"], str)
            signature = block["extras"]["signature"]
            assert signature and isinstance(signature, str)

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message]):
        full = cast(BaseMessageChunk, chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert any("reasoning" in block for block in full.content)
    for block in full.content:
        assert isinstance(block, dict)
        if block["type"] == "reasoning":
            assert set(block.keys()) == {"type", "reasoning", "extras", "index"}
            assert block["reasoning"] and isinstance(block["reasoning"], str)
            signature = block["extras"]["signature"]
            assert signature and isinstance(signature, str)

    # Test pass back in
    next_message = {"role": "user", "content": "How are you?"}
    _ = llm.invoke([input_message, full, next_message])


@pytest.mark.default_cassette("test_redacted_thinking.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_redacted_thinking(output_version: Literal["v0", "v1"]) -> None:
    llm = ChatAnthropic(
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
        elif block["type"] == "non_standard" and block["value"]["type"] == "redacted_thinking":
            value = block["value"]
        else:
            pass
        if value:
            assert set(value.keys()) == {"type", "data"}
            assert value["data"] and isinstance(value["data"], str)
    assert value is not None

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message]):
        full = cast(BaseMessageChunk, chunk) if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    value = None
    for block in full.content:
        assert isinstance(block, dict)
        if block["type"] == "redacted_thinking":
            value = block
            assert set(value.keys()) == {"type", "data", "index"}
            assert "index" in block
        elif block["type"] == "non_standard" and block["value"]["type"] == "redacted_thinking":
            value = block["value"]
            assert set(value.keys()) == {"type", "data"}
            assert "index" in block
        else:
            pass
        if value:
            assert value["data"] and isinstance(value["data"], str)
    assert value is not None

    # Test pass back in
    next_message = {"role": "user", "content": "What?"}
    _ = llm.invoke([input_message, full, next_message])


def test_structured_output_thinking_enabled() -> None:
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-latest",  # type: ignore[call-arg]
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
        model="claude-3-7-sonnet-latest",  # type: ignore[call-arg]
        max_tokens=5_000,  # type: ignore[call-arg]
        thinking={"type": "enabled", "budget_tokens": 2_000},
    ).bind_tools(
        [GenerateUsername],
        tool_choice="GenerateUsername",
    )
    with pytest.raises(BadRequestError):
        llm.invoke("Generate a username for Sally with green hair")


def test_image_tool_calling() -> None:
    """Test tool calling with image inputs."""

    class color_picker(BaseModel):
        """Input your fav color and get a random fact about it."""

        fav_color: str

    human_content: list[dict] = [
        {
            "type": "text",
            "text": "what's your favorite color in this image",
        },
    ]
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    image_data = b64encode(httpx.get(image_url).content).decode("utf-8")
    human_content.append(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
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
                    "input": {"fav_color": "green"},
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
                            "text": "green is a great pick! that's my sister's favorite color",  # noqa: E501
                        },
                    ],
                    "is_error": False,
                },
                {"type": "text", "text": "what's my sister's favorite color"},
            ],
        ),
    ]
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest")  # type: ignore[call-arg]
    llm.bind_tools([color_picker]).invoke(messages)


@pytest.mark.default_cassette("test_web_search.yaml.gz")
@pytest.mark.vcr
@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_web_search(output_version: Literal["v0", "v1"]) -> None:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",  # type: ignore[call-arg]
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
        assert block_types == {"text", "web_search_call", "web_search_result"}

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    if output_version == "v0":
        assert block_types == {"text", "server_tool_use", "web_search_tool_result"}
    else:
        assert block_types == {"text", "web_search_call", "web_search_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please repeat the last search, but focus on sources from 2024.",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )


@pytest.mark.vcr
def test_code_execution() -> None:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",  # type: ignore[call-arg]
        betas=["code-execution-2025-05-22"],
        max_tokens=10_000,  # type: ignore[call-arg]
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
    assert block_types == {"text", "server_tool_use", "code_execution_tool_result"}

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    assert block_types == {"text", "server_tool_use", "code_execution_tool_result"}

    # Test we can pass back in
    next_message = {
        "role": "user",
        "content": "Please add more comments to the code.",
    }
    _ = llm_with_tools.invoke(
        [input_message, full, next_message],
    )


@pytest.mark.vcr
def test_remote_mcp() -> None:
    mcp_servers = [
        {
            "type": "url",
            "url": "https://mcp.deepwiki.com/mcp",
            "name": "deepwiki",
            "tool_configuration": {"enabled": True, "allowed_tools": ["ask_question"]},
            "authorization_token": "PLACEHOLDER",
        },
    ]

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",  # type: ignore[call-arg]
        betas=["mcp-client-2025-04-04"],
        mcp_servers=mcp_servers,
        max_tokens=10_000,  # type: ignore[call-arg]
    )

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
    assert block_types == {"text", "mcp_tool_use", "mcp_tool_result"}

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert all(isinstance(block, dict) for block in full.content)
    block_types = {block["type"] for block in full.content}  # type: ignore[index]
    assert block_types == {"text", "mcp_tool_use", "mcp_tool_result"}

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
    image_file_id = os.getenv("ANTHROPIC_FILES_API_IMAGE_ID")
    if not image_file_id:
        pytest.skip()
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",  # type: ignore[call-arg]
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
            "source_type": "id",
            "id": image_file_id,
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
    pdf_file_id = os.getenv("ANTHROPIC_FILES_API_PDF_ID")
    if not pdf_file_id:
        pytest.skip()
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",  # type: ignore[call-arg]
        betas=["files-api-2025-04-14"],
    )
    if block_format == "anthropic":
        block = {"type": "document", "source": {"type": "file", "file_id": pdf_file_id}}
    else:
        # standard block format
        block = {
            "type": "file",
            "source_type": "id",
            "id": pdf_file_id,
        }
    input_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this document."},
            block,
        ],
    }
    _ = llm.invoke([input_message])


def test_search_result_tool_message() -> None:
    """Test that we can pass a search result tool message to the model."""
    llm = ChatAnthropic(
        model="claude-3-5-haiku-latest",  # type: ignore[call-arg]
        betas=["search-results-2025-06-09"],
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


def test_search_result_top_level() -> None:
    llm = ChatAnthropic(
        model="claude-3-5-haiku-latest",  # type: ignore[call-arg]
        betas=["search-results-2025-06-09"],
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


def test_async_shared_client() -> None:
    llm = ChatAnthropic(model="claude-3-5-haiku-latest")  # type: ignore[call-arg]
    _ = asyncio.run(llm.ainvoke("Hello"))
    _ = asyncio.run(llm.ainvoke("Hello"))

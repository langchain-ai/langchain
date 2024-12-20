"""Test ChatOpenAI chat model."""

import base64
import json
from pathlib import Path
from textwrap import dedent
from typing import Any, AsyncIterator, List, Literal, Optional, cast

import httpx
import openai
import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_tests.integration_tests.chat_models import _validate_tool_call_message
from langchain_tests.integration_tests.chat_models import (
    magic_function as invalid_magic_function,
)
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from tests.unit_tests.fake.callbacks import FakeCallbackHandler


@pytest.mark.scheduled
def test_chat_openai() -> None:
    """Test ChatOpenAI wrapper."""
    chat = ChatOpenAI(
        temperature=0.7,
        base_url=None,
        organization=None,
        openai_proxy=None,
        timeout=10.0,
        max_retries=3,
        http_client=None,
        n=1,
        max_completion_tokens=10,
        default_headers=None,
        default_query=None,
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_openai_model() -> None:
    """Test ChatOpenAI wrapper handles model_name."""
    chat = ChatOpenAI(model="foo")
    assert chat.model_name == "foo"
    chat = ChatOpenAI(model_name="bar")  # type: ignore[call-arg]
    assert chat.model_name == "bar"


def test_chat_openai_system_message() -> None:
    """Test ChatOpenAI wrapper with system message."""
    chat = ChatOpenAI(max_completion_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_openai_generate() -> None:
    """Test ChatOpenAI wrapper with generate."""
    chat = ChatOpenAI(max_completion_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
def test_chat_openai_multiple_completions() -> None:
    """Test ChatOpenAI wrapper with multiple completions."""
    chat = ChatOpenAI(max_completion_tokens=10, n=5)
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


@pytest.mark.scheduled
def test_chat_openai_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatOpenAI(
        max_completion_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_chat_openai_streaming_generation_info() -> None:
    """Test that generation info is preserved when streaming."""

    class _FakeCallback(FakeCallbackHandler):
        saved_things: dict = {}

        def on_llm_end(self, *args: Any, **kwargs: Any) -> Any:
            # Save the generation
            self.saved_things["generation"] = args[0]

    callback = _FakeCallback()
    callback_manager = CallbackManager([callback])
    chat = ChatOpenAI(
        max_completion_tokens=2, temperature=0, callback_manager=callback_manager
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello!"


def test_chat_openai_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_completion_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_openai_streaming_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatOpenAI(max_completion_tokens=10, streaming=True)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_name


def test_chat_openai_invalid_streaming_params() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with pytest.raises(ValueError):
        ChatOpenAI(max_completion_tokens=10, streaming=True, temperature=0, n=5)


@pytest.mark.scheduled
async def test_async_chat_openai() -> None:
    """Test async generation."""
    chat = ChatOpenAI(max_completion_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
async def test_async_chat_openai_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatOpenAI(
        max_completion_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
async def test_async_chat_openai_bind_functions() -> None:
    """Test ChatOpenAI wrapper with multiple completions."""

    class Person(BaseModel):
        """Identifying information about a person."""

        name: str = Field(..., title="Name", description="The person's name")
        age: int = Field(..., title="Age", description="The person's age")
        fav_food: Optional[str] = Field(
            default=None, title="Fav Food", description="The person's favorite food"
        )

    chat = ChatOpenAI(max_completion_tokens=30, n=1, streaming=True).bind_functions(
        functions=[Person], function_call="Person"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", "Use the provided Person function"), ("user", "{input}")]
    )

    chain = prompt | chat

    message = HumanMessage(content="Sally is 13 years old")
    response = await chain.abatch([{"input": message}])

    assert isinstance(response, list)
    assert len(response) == 1
    for generation in response:
        assert isinstance(generation, AIMessage)


@pytest.mark.scheduled
def test_openai_streaming() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI(max_completion_tokens=10)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_astream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI(max_completion_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_abatch() -> None:
    """Test streaming tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_completion_tokens=10)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_abatch_tags() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_completion_tokens=10)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_openai_batch() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_completion_tokens=10)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_ainvoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_completion_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_openai_invoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI(max_completion_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)

    # assert no response headers if include_response_headers is not set
    assert "headers" not in result.response_metadata


def test_stream() -> None:
    """Test streaming tokens from OpenAI."""
    llm = ChatOpenAI()

    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("I'm Pickle Rick"):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata.get("finish_reason") is not None
    assert full.response_metadata.get("model_name") is not None

    # check token usage
    aggregate: Optional[BaseMessageChunk] = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    for chunk in llm.stream("Hello", stream_usage=True):
        assert isinstance(chunk.content, str)
        aggregate = chunk if aggregate is None else aggregate + chunk
        assert isinstance(chunk, AIMessageChunk)
        if chunk.usage_metadata is not None:
            chunks_with_token_counts += 1
        if chunk.response_metadata:
            chunks_with_response_metadata += 1
    if chunks_with_token_counts != 1 or chunks_with_response_metadata != 1:
        raise AssertionError(
            "Expected exactly one chunk with metadata. "
            "AIMessageChunk aggregation can add these metadata. Check that "
            "this is behaving properly."
        )
    assert isinstance(aggregate, AIMessageChunk)
    assert aggregate.usage_metadata is not None
    assert aggregate.usage_metadata["input_tokens"] > 0
    assert aggregate.usage_metadata["output_tokens"] > 0
    assert aggregate.usage_metadata["total_tokens"] > 0


async def test_astream() -> None:
    """Test streaming tokens from OpenAI."""

    async def _test_stream(stream: AsyncIterator, expect_usage: bool) -> None:
        full: Optional[BaseMessageChunk] = None
        chunks_with_token_counts = 0
        chunks_with_response_metadata = 0
        async for chunk in stream:
            assert isinstance(chunk.content, str)
            full = chunk if full is None else full + chunk
            assert isinstance(chunk, AIMessageChunk)
            if chunk.usage_metadata is not None:
                chunks_with_token_counts += 1
            if chunk.response_metadata:
                chunks_with_response_metadata += 1
        assert isinstance(full, AIMessageChunk)
        if chunks_with_response_metadata != 1:
            raise AssertionError(
                "Expected exactly one chunk with metadata. "
                "AIMessageChunk aggregation can add these metadata. Check that "
                "this is behaving properly."
            )
        assert full.response_metadata.get("finish_reason") is not None
        assert full.response_metadata.get("model_name") is not None
        if expect_usage:
            if chunks_with_token_counts != 1:
                raise AssertionError(
                    "Expected exactly one chunk with token counts. "
                    "AIMessageChunk aggregation adds counts. Check that "
                    "this is behaving properly."
                )
            assert full.usage_metadata is not None
            assert full.usage_metadata["input_tokens"] > 0
            assert full.usage_metadata["output_tokens"] > 0
            assert full.usage_metadata["total_tokens"] > 0
        else:
            assert chunks_with_token_counts == 0
            assert full.usage_metadata is None

    llm = ChatOpenAI(temperature=0, max_completion_tokens=5)
    await _test_stream(llm.astream("Hello"), expect_usage=False)
    await _test_stream(
        llm.astream("Hello", stream_options={"include_usage": True}), expect_usage=True
    )
    await _test_stream(llm.astream("Hello", stream_usage=True), expect_usage=True)
    llm = ChatOpenAI(
        temperature=0,
        max_completion_tokens=5,
        model_kwargs={"stream_options": {"include_usage": True}},
    )
    await _test_stream(llm.astream("Hello"), expect_usage=True)
    await _test_stream(
        llm.astream("Hello", stream_options={"include_usage": False}),
        expect_usage=False,
    )
    llm = ChatOpenAI(temperature=0, max_completion_tokens=5, stream_usage=True)
    await _test_stream(llm.astream("Hello"), expect_usage=True)
    await _test_stream(llm.astream("Hello", stream_usage=False), expect_usage=False)


async def test_abatch() -> None:
    """Test streaming tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_abatch_tags() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_batch() -> None:
    """Test batch tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)
    assert result.response_metadata.get("model_name") is not None


def test_invoke() -> None:
    """Test invoke tokens from ChatOpenAI."""
    llm = ChatOpenAI()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
    assert result.response_metadata.get("model_name") is not None


def test_response_metadata() -> None:
    llm = ChatOpenAI()
    result = llm.invoke([HumanMessage(content="I'm PickleRick")], logprobs=True)
    assert result.response_metadata
    assert all(
        k in result.response_metadata
        for k in (
            "token_usage",
            "model_name",
            "logprobs",
            "system_fingerprint",
            "finish_reason",
        )
    )
    assert "content" in result.response_metadata["logprobs"]


async def test_async_response_metadata() -> None:
    llm = ChatOpenAI()
    result = await llm.ainvoke([HumanMessage(content="I'm PickleRick")], logprobs=True)
    assert result.response_metadata
    assert all(
        k in result.response_metadata
        for k in (
            "token_usage",
            "model_name",
            "logprobs",
            "system_fingerprint",
            "finish_reason",
        )
    )
    assert "content" in result.response_metadata["logprobs"]


def test_response_metadata_streaming() -> None:
    llm = ChatOpenAI()
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("I'm Pickle Rick", logprobs=True):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert all(
        k in cast(BaseMessageChunk, full).response_metadata
        for k in ("logprobs", "finish_reason")
    )
    assert "content" in cast(BaseMessageChunk, full).response_metadata["logprobs"]


async def test_async_response_metadata_streaming() -> None:
    llm = ChatOpenAI()
    full: Optional[BaseMessageChunk] = None
    async for chunk in llm.astream("I'm Pickle Rick", logprobs=True):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert all(
        k in cast(BaseMessageChunk, full).response_metadata
        for k in ("logprobs", "finish_reason")
    )
    assert "content" in cast(BaseMessageChunk, full).response_metadata["logprobs"]


class GenerateUsername(BaseModel):
    "Get a username based on someone's name and hair color."

    name: str
    hair_color: str


class MakeASandwich(BaseModel):
    "Make a sandwich given a list of ingredients."

    bread_type: str
    cheese_type: str
    condiments: List[str]
    vegetables: List[str]


def test_tool_use() -> None:
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    llm_with_tool = llm.bind_tools(tools=[GenerateUsername], tool_choice=True)
    msgs: List = [HumanMessage("Sally has green hair, what would her username be?")]
    ai_msg = llm_with_tool.invoke(msgs)

    assert isinstance(ai_msg, AIMessage)
    assert isinstance(ai_msg.tool_calls, list)
    assert len(ai_msg.tool_calls) == 1
    tool_call = ai_msg.tool_calls[0]
    assert "args" in tool_call

    tool_msg = ToolMessage(
        "sally_green_hair", tool_call_id=ai_msg.additional_kwargs["tool_calls"][0]["id"]
    )
    msgs.extend([ai_msg, tool_msg])
    llm_with_tool.invoke(msgs)

    # Test streaming
    ai_messages = llm_with_tool.stream(msgs)
    first = True
    for message in ai_messages:
        if first:
            gathered = message
            first = False
        else:
            gathered = gathered + message  # type: ignore
    assert isinstance(gathered, AIMessageChunk)
    assert isinstance(gathered.tool_call_chunks, list)
    assert len(gathered.tool_call_chunks) == 1
    tool_call_chunk = gathered.tool_call_chunks[0]
    assert "args" in tool_call_chunk

    streaming_tool_msg = ToolMessage(
        "sally_green_hair",
        tool_call_id=gathered.additional_kwargs["tool_calls"][0]["id"],
    )
    msgs.extend([gathered, streaming_tool_msg])
    llm_with_tool.invoke(msgs)


def test_manual_tool_call_msg() -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm_with_tool = llm.bind_tools(tools=[GenerateUsername])
    msgs: List = [
        HumanMessage("Sally has green hair, what would her username be?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="GenerateUsername",
                    args={"name": "Sally", "hair_color": "green"},
                    id="foo",
                )
            ],
        ),
        ToolMessage("sally_green_hair", tool_call_id="foo"),
    ]
    output: AIMessage = cast(AIMessage, llm_with_tool.invoke(msgs))
    assert output.content
    # Should not have called the tool again.
    assert not output.tool_calls and not output.invalid_tool_calls

    # OpenAI should error when tool call id doesn't match across AIMessage and
    # ToolMessage
    msgs = [
        HumanMessage("Sally has green hair, what would her username be?"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="GenerateUsername",
                    args={"name": "Sally", "hair_color": "green"},
                    id="bar",
                )
            ],
        ),
        ToolMessage("sally_green_hair", tool_call_id="foo"),
    ]
    with pytest.raises(Exception):
        llm_with_tool.invoke(msgs)


def test_bind_tools_tool_choice() -> None:
    """Test passing in manually construct tool call message."""
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    for tool_choice in ("any", "required"):
        llm_with_tools = llm.bind_tools(
            tools=[GenerateUsername, MakeASandwich], tool_choice=tool_choice
        )
        msg = cast(AIMessage, llm_with_tools.invoke("how are you"))
        assert msg.tool_calls

    llm_with_tools = llm.bind_tools(tools=[GenerateUsername, MakeASandwich])
    msg = cast(AIMessage, llm_with_tools.invoke("how are you"))
    assert not msg.tool_calls


def test_openai_structured_output() -> None:
    class MyModel(BaseModel):
        """A Person"""

        name: str
        age: int

    llm = ChatOpenAI().with_structured_output(MyModel)
    result = llm.invoke("I'm a 27 year old named Erick")
    assert isinstance(result, MyModel)
    assert result.name == "Erick"
    assert result.age == 27


def test_openai_proxy() -> None:
    """Test ChatOpenAI with proxy."""
    chat_openai = ChatOpenAI(openai_proxy="http://localhost:8080")
    mounts = chat_openai.client._client._client._mounts
    assert len(mounts) == 1
    for key, value in mounts.items():
        proxy = value._pool._proxy_url.origin
        assert proxy.scheme == b"http"
        assert proxy.host == b"localhost"
        assert proxy.port == 8080

    async_client_mounts = chat_openai.async_client._client._client._mounts
    assert len(async_client_mounts) == 1
    for key, value in async_client_mounts.items():
        proxy = value._pool._proxy_url.origin
        assert proxy.scheme == b"http"
        assert proxy.host == b"localhost"
        assert proxy.port == 8080


def test_openai_response_headers() -> None:
    """Test ChatOpenAI response headers."""
    chat_openai = ChatOpenAI(include_response_headers=True)
    query = "I'm Pickle Rick"
    result = chat_openai.invoke(query, max_completion_tokens=10)
    headers = result.response_metadata["headers"]
    assert headers
    assert isinstance(headers, dict)
    assert "content-type" in headers

    # Stream
    full: Optional[BaseMessageChunk] = None
    for chunk in chat_openai.stream(query, max_completion_tokens=10):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessage)
    headers = full.response_metadata["headers"]
    assert headers
    assert isinstance(headers, dict)
    assert "content-type" in headers


async def test_openai_response_headers_async() -> None:
    """Test ChatOpenAI response headers."""
    chat_openai = ChatOpenAI(include_response_headers=True)
    query = "I'm Pickle Rick"
    result = await chat_openai.ainvoke(query, max_completion_tokens=10)
    headers = result.response_metadata["headers"]
    assert headers
    assert isinstance(headers, dict)
    assert "content-type" in headers

    # Stream
    full: Optional[BaseMessageChunk] = None
    async for chunk in chat_openai.astream(query, max_completion_tokens=10):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessage)
    headers = full.response_metadata["headers"]
    assert headers
    assert isinstance(headers, dict)
    assert "content-type" in headers


@pytest.mark.xfail(
    reason=(
        "As of 12.19.24 OpenAI API returns 1151 instead of 1118. Not clear yet if "
        "this is an undocumented API change or a bug on their end."
    )
)
def test_image_token_counting_jpeg() -> None:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe the weather in this image"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
        "input_tokens"
    ]
    actual = model.get_num_tokens_from_messages([message])
    assert expected == actual

    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "describe the weather in this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ]
    )
    expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
        "input_tokens"
    ]
    actual = model.get_num_tokens_from_messages([message])
    assert expected == actual


@pytest.mark.xfail(
    reason=(
        "As of 12.19.24 OpenAI API returns 871 instead of 779. Not clear yet if "
        "this is an undocumented API change or a bug on their end."
    )
)
def test_image_token_counting_png() -> None:
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
    message = HumanMessage(
        content=[
            {"type": "text", "text": "how many dice are in this image"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
        "input_tokens"
    ]
    actual = model.get_num_tokens_from_messages([message])
    assert expected == actual

    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "how many dice are in this image"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
    )
    expected = cast(AIMessage, model.invoke([message])).usage_metadata[  # type: ignore[index]
        "input_tokens"
    ]
    actual = model.get_num_tokens_from_messages([message])
    assert expected == actual


def test_tool_calling_strict() -> None:
    """Test tool calling with strict=True."""

    class magic_function(BaseModel):
        """Applies a magic function to an input."""

        input: int

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    model_with_tools = model.bind_tools([magic_function], strict=True)

    # invalid_magic_function adds metadata to schema that isn't supported by OpenAI.
    model_with_invalid_tool_schema = model.bind_tools(
        [invalid_magic_function], strict=True
    )

    # Test invoke
    query = "What is the value of magic_function(3)? Use the tool."
    response = model_with_tools.invoke(query)
    _validate_tool_call_message(response)

    # Test invalid tool schema
    with pytest.raises(openai.BadRequestError):
        model_with_invalid_tool_schema.invoke(query)

    # Test stream
    full: Optional[BaseMessageChunk] = None
    for chunk in model_with_tools.stream(query):
        full = chunk if full is None else full + chunk  # type: ignore
    assert isinstance(full, AIMessage)
    _validate_tool_call_message(full)

    # Test invalid tool schema
    with pytest.raises(openai.BadRequestError):
        next(model_with_invalid_tool_schema.stream(query))


@pytest.mark.parametrize(
    ("model", "method", "strict"),
    [("gpt-4o", "function_calling", True), ("gpt-4o-2024-08-06", "json_schema", None)],
)
def test_structured_output_strict(
    model: str,
    method: Literal["function_calling", "json_schema"],
    strict: Optional[bool],
) -> None:
    """Test to verify structured output with strict=True."""

    from pydantic import BaseModel as BaseModelProper
    from pydantic import Field as FieldProper

    llm = ChatOpenAI(model=model, temperature=0)

    class Joke(BaseModelProper):
        """Joke to tell user."""

        setup: str = FieldProper(description="question to set up a joke")
        punchline: str = FieldProper(description="answer to resolve the joke")

    # Pydantic class
    # Type ignoring since the interface only officially supports pydantic 1
    # or pydantic.v1.BaseModel but not pydantic.BaseModel from pydantic 2.
    # We'll need to do a pass updating the type signatures.
    chat = llm.with_structured_output(Joke, method=method, strict=strict)
    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, Joke)

    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, Joke)

    # Schema
    chat = llm.with_structured_output(
        Joke.model_json_schema(), method=method, strict=strict
    )
    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"setup", "punchline"}

    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, dict)
    assert isinstance(chunk, dict)  # for mypy
    assert set(chunk.keys()) == {"setup", "punchline"}

    # Invalid schema with optional fields:
    class InvalidJoke(BaseModelProper):
        """Joke to tell user."""

        setup: str = FieldProper(description="question to set up a joke")
        # Invalid field, can't have default value.
        punchline: str = FieldProper(
            default="foo", description="answer to resolve the joke"
        )

    chat = llm.with_structured_output(InvalidJoke, method=method, strict=strict)
    with pytest.raises(openai.BadRequestError):
        chat.invoke("Tell me a joke about cats.")
    with pytest.raises(openai.BadRequestError):
        next(chat.stream("Tell me a joke about cats."))

    chat = llm.with_structured_output(
        InvalidJoke.model_json_schema(), method=method, strict=strict
    )
    with pytest.raises(openai.BadRequestError):
        chat.invoke("Tell me a joke about cats.")
    with pytest.raises(openai.BadRequestError):
        next(chat.stream("Tell me a joke about cats."))


@pytest.mark.parametrize(
    ("model", "method", "strict"), [("gpt-4o-2024-08-06", "json_schema", None)]
)
def test_nested_structured_output_strict(
    model: str, method: Literal["json_schema"], strict: Optional[bool]
) -> None:
    """Test to verify structured output with strict=True for nested object."""

    from typing import TypedDict

    llm = ChatOpenAI(model=model, temperature=0)

    class SelfEvaluation(TypedDict):
        score: int
        text: str

    class JokeWithEvaluation(TypedDict):
        """Joke to tell user."""

        setup: str
        punchline: str
        self_evaluation: SelfEvaluation

    # Schema
    chat = llm.with_structured_output(JokeWithEvaluation, method=method, strict=strict)
    result = chat.invoke("Tell me a joke about cats.")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"setup", "punchline", "self_evaluation"}
    assert set(result["self_evaluation"].keys()) == {"score", "text"}

    for chunk in chat.stream("Tell me a joke about cats."):
        assert isinstance(chunk, dict)
    assert isinstance(chunk, dict)  # for mypy
    assert set(chunk.keys()) == {"setup", "punchline", "self_evaluation"}
    assert set(chunk["self_evaluation"].keys()) == {"score", "text"}


def test_json_mode() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(
        "Return this as json: {'a': 1}. Do not return anything other than json. Do not include markdown codeblocks.",  # noqa: E501
        response_format={"type": "json_object"},
    )
    assert isinstance(response.content, str)
    assert json.loads(response.content) == {"a": 1}

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream(
        "Return this as json: {'a': 1}", response_format={"type": "json_object"}
    ):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, str)
    assert json.loads(full.content) == {"a": 1}


async def test_json_mode_async() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = await llm.ainvoke(
        "Return this as json: {'a': 1}. Do not return anything other than json. Do not include markdown codeblocks."  # noqa: E501
    )
    assert isinstance(response.content, str)
    assert json.loads(response.content) == {"a": 1}

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    async for chunk in llm.astream(
        "Return this as json: {'a': 1}", response_format={"type": "json_object"}
    ):
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, str)
    assert json.loads(full.content) == {"a": 1}


def test_audio_output_modality() -> None:
    llm = ChatOpenAI(
        model="gpt-4o-audio-preview",
        temperature=0,
        model_kwargs={
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
        },
    )

    history: List[BaseMessage] = [
        HumanMessage("Make me a short audio clip of you yelling")
    ]

    output = llm.invoke(history)

    assert isinstance(output, AIMessage)
    assert "audio" in output.additional_kwargs

    history.append(output)
    history.append(HumanMessage("Make me a short audio clip of you whispering"))

    output = llm.invoke(history)

    assert isinstance(output, AIMessage)
    assert "audio" in output.additional_kwargs


def test_audio_input_modality() -> None:
    llm = ChatOpenAI(
        model="gpt-4o-audio-preview",
        temperature=0,
        model_kwargs={
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
        },
    )
    filepath = Path(__file__).parent / "audio_input.wav"

    audio_data = filepath.read_bytes()
    b64_audio_data = base64.b64encode(audio_data).decode("utf-8")

    history: list[BaseMessage] = [
        HumanMessage(
            [
                {"type": "text", "text": "What is happening in this audio clip"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": b64_audio_data, "format": "wav"},
                },
            ]
        )
    ]

    output = llm.invoke(history)

    assert isinstance(output, AIMessage)
    assert "audio" in output.additional_kwargs

    history.append(output)
    history.append(HumanMessage("Why?"))

    output = llm.invoke(history)

    assert isinstance(output, AIMessage)
    assert "audio" in output.additional_kwargs


def test_prediction_tokens() -> None:
    code = dedent(
        """
    /// <summary>
    /// Represents a user with a first name, last name, and username.
    /// </summary>
    public class User
    {
        /// <summary>
        /// Gets or sets the user's first name.
        /// </summary>
        public string FirstName { get; set; }

        /// <summary>
        /// Gets or sets the user's last name.
        /// </summary>
        public string LastName { get; set; }

        /// <summary>
        /// Gets or sets the user's username.
        /// </summary>
        public string Username { get; set; }
    }
    """
    )

    llm = ChatOpenAI(model="gpt-4o")
    query = (
        "Replace the Username property with an Email property. "
        "Respond only with code, and with no markdown formatting."
    )
    response = llm.invoke(
        [{"role": "user", "content": query}, {"role": "user", "content": code}],
        prediction={"type": "content", "content": code},
    )
    assert isinstance(response, AIMessage)
    assert response.response_metadata is not None
    output_token_details = response.response_metadata["token_usage"][
        "completion_tokens_details"
    ]
    assert output_token_details["accepted_prediction_tokens"] > 0
    assert output_token_details["rejected_prediction_tokens"] > 0


def test_stream_o1() -> None:
    list(ChatOpenAI(model="o1-mini").stream("how are you"))


async def test_astream_o1() -> None:
    async for _ in ChatOpenAI(model="o1-mini").astream("how are you"):
        pass


class Foo(BaseModel):
    response: str


def test_stream_response_format() -> None:
    list(ChatOpenAI(model="gpt-4o-mini").stream("how are ya", response_format=Foo))


async def test_astream_response_format() -> None:
    async for _ in ChatOpenAI(model="gpt-4o-mini").astream(
        "how are ya", response_format=Foo
    ):
        pass


@pytest.mark.parametrize("use_max_completion_tokens", [True, False])
def test_o1(use_max_completion_tokens: bool) -> None:
    if use_max_completion_tokens:
        kwargs: dict = {"max_completion_tokens": 10}
    else:
        kwargs = {"max_tokens": 10}
    response = ChatOpenAI(model="o1", reasoning_effort="low", **kwargs).invoke(
        [
            {"role": "developer", "content": "respond in all caps"},
            {"role": "user", "content": "HOW ARE YOU"},
        ]
    )
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content.upper() == response.content

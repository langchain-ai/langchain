"""Test ChatGroq chat model."""

from __future__ import annotations

import json
from typing import Any, Optional, cast

import pytest
from groq import BadRequestError
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from tests.unit_tests.fake.callbacks import (
    FakeCallbackHandler,
    FakeCallbackHandlerWithChatStart,
)

DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"
REASONING_MODEL_NAME = "deepseek-r1-distill-llama-70b"


#
# Smoke test Runnable interface
#
@pytest.mark.scheduled
def test_invoke() -> None:
    """Test Chat wrapper."""
    chat = ChatGroq(
        model=DEFAULT_MODEL_NAME,
        temperature=0.7,
        base_url=None,
        groq_proxy=None,
        timeout=10.0,
        max_retries=3,
        http_client=None,
        n=1,
        max_tokens=10,
        default_headers=None,
        default_query=None,
    )
    message = HumanMessage(content="Welcome to the Groqetship")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
async def test_ainvoke() -> None:
    """Test ainvoke tokens from ChatGroq."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)

    result = await chat.ainvoke("Welcome to the Groqetship!", config={"tags": ["foo"]})
    assert isinstance(result, BaseMessage)
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_batch() -> None:
    """Test batch tokens from ChatGroq."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)

    result = chat.batch(["Hello!", "Welcome to the Groqetship!"])
    for token in result:
        assert isinstance(token, BaseMessage)
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_abatch() -> None:
    """Test abatch tokens from ChatGroq."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)

    result = await chat.abatch(["Hello!", "Welcome to the Groqetship!"])
    for token in result:
        assert isinstance(token, BaseMessage)
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_stream() -> None:
    """Test streaming tokens from Groq."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)

    for token in chat.stream("Welcome to the Groqetship!"):
        assert isinstance(token, BaseMessageChunk)
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_astream() -> None:
    """Test streaming tokens from Groq."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)

    full: Optional[BaseMessageChunk] = None
    chunks_with_token_counts = 0
    chunks_with_response_metadata = 0
    async for token in chat.astream("Welcome to the Groqetship!"):
        assert isinstance(token, AIMessageChunk)
        assert isinstance(token.content, str)
        full = token if full is None else full + token
        if token.usage_metadata is not None:
            chunks_with_token_counts += 1
        if token.response_metadata:
            chunks_with_response_metadata += 1
    if chunks_with_token_counts != 1 or chunks_with_response_metadata != 1:
        msg = (
            "Expected exactly one chunk with token counts or metadata. "
            "AIMessageChunk aggregation adds / appends these metadata. Check that "
            "this is behaving properly."
        )
        raise AssertionError(msg)
    assert isinstance(full, AIMessageChunk)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert (
        full.usage_metadata["input_tokens"] + full.usage_metadata["output_tokens"]
        == full.usage_metadata["total_tokens"]
    )
    for expected_metadata in ["model_name", "system_fingerprint"]:
        assert full.response_metadata[expected_metadata]


#
# Test Legacy generate methods
#
@pytest.mark.scheduled
def test_generate() -> None:
    """Test sync generate."""
    n = 1
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)
    message = HumanMessage(content="Hello", n=1)
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    assert response.llm_output["model_name"] == chat.model_name
    for generations in response.generations:
        assert len(generations) == n
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
async def test_agenerate() -> None:
    """Test async generation."""
    n = 1
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10, n=1)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    assert response.llm_output["model_name"] == chat.model_name
    for generations in response.generations:
        assert len(generations) == n
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


#
# Test streaming flags in invoke and generate
#
@pytest.mark.scheduled
def test_invoke_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    chat = ChatGroq(
        model=DEFAULT_MODEL_NAME,
        max_tokens=2,
        streaming=True,
        temperature=0,
        callbacks=[callback_handler],
    )
    message = HumanMessage(content="Welcome to the Groqetship")
    response = chat.invoke([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
async def test_agenerate_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandlerWithChatStart()
    chat = ChatGroq(
        model=DEFAULT_MODEL_NAME,
        max_tokens=10,
        streaming=True,
        temperature=0,
        callbacks=[callback_handler],
    )
    message = HumanMessage(content="Welcome to the Groqetship")
    response = await chat.agenerate([[message], [message]])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output is not None
    assert response.llm_output["model_name"] == chat.model_name
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


#
# Test reasoning output
#
def test_reasoning_output_invoke() -> None:
    """Test reasoning output from ChatGroq with invoke."""
    chat = ChatGroq(
        model=REASONING_MODEL_NAME,
        reasoning_format="parsed",
    )
    message = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love programming."),
    ]
    response = chat.invoke(message)
    assert isinstance(response, AIMessage)
    assert "reasoning_content" in response.additional_kwargs
    assert isinstance(response.additional_kwargs["reasoning_content"], str)
    assert len(response.additional_kwargs["reasoning_content"]) > 0


def test_reasoning_output_stream() -> None:
    """Test reasoning output from ChatGroq with stream."""
    chat = ChatGroq(
        model=REASONING_MODEL_NAME,
        reasoning_format="parsed",
    )
    message = [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love programming."),
    ]

    full_response: Optional[AIMessageChunk] = None
    for token in chat.stream(message):
        assert isinstance(token, AIMessageChunk)

        if full_response is None:
            full_response = token
        else:
            # Casting since adding results in a type error
            full_response = cast(AIMessageChunk, full_response + token)

    assert full_response is not None
    assert isinstance(full_response, AIMessageChunk)
    assert "reasoning_content" in full_response.additional_kwargs
    assert isinstance(full_response.additional_kwargs["reasoning_content"], str)
    assert len(full_response.additional_kwargs["reasoning_content"]) > 0


def test_reasoning_effort_none() -> None:
    """Test that no reasoning output is returned if effort is set to none."""
    chat = ChatGroq(
        model="qwen/qwen3-32b",  # Only qwen3 currently supports reasoning_effort
        reasoning_effort="none",
    )
    message = HumanMessage(content="What is the capital of France?")
    response = chat.invoke([message])
    assert isinstance(response, AIMessage)
    assert "reasoning_content" not in response.additional_kwargs
    assert "<think>" not in response.content and "<think/>" not in response.content


#
# Misc tests
#
def test_streaming_generation_info() -> None:
    """Test that generation info is preserved when streaming."""

    class _FakeCallback(FakeCallbackHandler):
        saved_things: dict = {}

        def on_llm_end(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Save the generation
            self.saved_things["generation"] = args[0]

    callback = _FakeCallback()
    chat = ChatGroq(
        model=DEFAULT_MODEL_NAME,
        max_tokens=2,
        temperature=0,
        callbacks=[callback],
    )
    list(chat.stream("Respond with the single word Hello", stop=["o"]))
    generation = callback.saved_things["generation"]
    # Verify that generation info is preserved when streaming
    assert isinstance(generation, LLMResult)

    # The generation should exist even if content is empty
    assert len(generation.generations) == 1
    assert len(generation.generations[0]) == 1

    # Generation info should be preserved
    gen = generation.generations[0][0]
    assert gen.generation_info is not None
    assert "finish_reason" in gen.generation_info
    assert "model_name" in gen.generation_info

    # For models that work properly, check the expected content
    # For models with empty streaming, just ensure structure is correct
    if gen.text:  # If content was returned, validate it
        assert gen.text == "Hell"
    else:  # If no content, ensure this is the problematic model behavior
        # At minimum, we should have completion tokens reported
        # The generation should be a ChatGeneration with an AIMessage
        from langchain_core.outputs import ChatGeneration

        if (
            isinstance(gen, ChatGeneration)
            and hasattr(gen.message, "usage_metadata")
            and gen.message.usage_metadata
        ):
            assert gen.message.usage_metadata.get("output_tokens", 0) > 0


def test_system_message() -> None:
    """Test ChatGroq wrapper with system message."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_tool_choice() -> None:
    """Test that tool choice is respected."""
    llm = ChatGroq(model=DEFAULT_MODEL_NAME)

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.invoke("Who was the 27 year old named Erick? Use the tool.")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"

    assert isinstance(resp.tool_calls, list)
    assert len(resp.tool_calls) == 1
    tool_call = resp.tool_calls[0]
    assert tool_call["name"] == "MyTool"
    assert tool_call["args"] == {"name": "Erick", "age": 27}


def test_tool_choice_bool() -> None:
    """Test that tool choice is respected just passing in True."""
    llm = ChatGroq(model=DEFAULT_MODEL_NAME)

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice=True)

    resp = with_tool.invoke("Who was the 27 year old named Erick? Use the tool.")
    assert isinstance(resp, AIMessage)
    assert resp.content == ""  # should just be tool call
    tool_calls = resp.additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"


@pytest.mark.xfail(reason="Groq tool_choice doesn't currently force a tool call")
def test_streaming_tool_call() -> None:
    """Test that tool choice is respected."""
    llm = ChatGroq(model=DEFAULT_MODEL_NAME)

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.stream("Who was the 27 year old named Erick?")
    additional_kwargs = None
    for chunk in resp:
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""  # should just be tool call
        additional_kwargs = chunk.additional_kwargs

    assert additional_kwargs is not None
    tool_calls = additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"

    assert isinstance(chunk, AIMessageChunk)
    assert isinstance(chunk.tool_call_chunks, list)
    assert len(chunk.tool_call_chunks) == 1
    tool_call_chunk = chunk.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "MyTool"
    assert isinstance(tool_call_chunk["args"], str)
    assert json.loads(tool_call_chunk["args"]) == {"name": "Erick", "age": 27}


@pytest.mark.xfail(reason="Groq tool_choice doesn't currently force a tool call")
async def test_astreaming_tool_call() -> None:
    """Test that tool choice is respected."""
    llm = ChatGroq(model=DEFAULT_MODEL_NAME)

    class MyTool(BaseModel):
        name: str
        age: int

    with_tool = llm.bind_tools([MyTool], tool_choice="MyTool")

    resp = with_tool.astream("Who was the 27 year old named Erick?")
    additional_kwargs = None
    async for chunk in resp:
        assert isinstance(chunk, AIMessageChunk)
        assert chunk.content == ""  # should just be tool call
        additional_kwargs = chunk.additional_kwargs

    assert additional_kwargs is not None
    tool_calls = additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == "MyTool"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "age": 27,
        "name": "Erick",
    }
    assert tool_call["type"] == "function"

    assert isinstance(chunk, AIMessageChunk)
    assert isinstance(chunk.tool_call_chunks, list)
    assert len(chunk.tool_call_chunks) == 1
    tool_call_chunk = chunk.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "MyTool"
    assert isinstance(tool_call_chunk["args"], str)
    assert json.loads(tool_call_chunk["args"]) == {"name": "Erick", "age": 27}


@pytest.mark.scheduled
def test_json_mode_structured_output() -> None:
    """Test with_structured_output with json."""

    class Joke(BaseModel):
        """Joke to tell user."""

        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

    chat = ChatGroq(model=DEFAULT_MODEL_NAME).with_structured_output(
        Joke, method="json_mode"
    )
    result = chat.invoke(
        "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
    )
    assert type(result) is Joke
    assert len(result.setup) != 0
    assert len(result.punchline) != 0


def test_setting_service_tier_class() -> None:
    """Test setting service tier defined at ChatGroq level."""
    message = HumanMessage(content="Welcome to the Groqetship")

    # Initialization
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="auto")
    assert chat.service_tier == "auto"
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
    assert response.response_metadata.get("service_tier") == "auto"

    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="flex")
    assert chat.service_tier == "flex"
    response = chat.invoke([message])
    assert response.response_metadata.get("service_tier") == "flex"

    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="on_demand")
    assert chat.service_tier == "on_demand"
    response = chat.invoke([message])
    assert response.response_metadata.get("service_tier") == "on_demand"

    chat = ChatGroq(model=DEFAULT_MODEL_NAME)
    assert chat.service_tier == "on_demand"
    response = chat.invoke([message])
    assert response.response_metadata.get("service_tier") == "on_demand"

    with pytest.raises(ValueError):
        ChatGroq(model=DEFAULT_MODEL_NAME, service_tier=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="invalid")  # type: ignore[arg-type]


def test_setting_service_tier_request() -> None:
    """Test setting service tier defined at request level."""
    message = HumanMessage(content="Welcome to the Groqetship")
    chat = ChatGroq(model=DEFAULT_MODEL_NAME)

    response = chat.invoke(
        [message],
        service_tier="auto",
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
    assert response.response_metadata.get("service_tier") == "auto"

    response = chat.invoke(
        [message],
        service_tier="flex",
    )
    assert response.response_metadata.get("service_tier") == "flex"

    response = chat.invoke(
        [message],
        service_tier="on_demand",
    )
    assert response.response_metadata.get("service_tier") == "on_demand"

    assert chat.service_tier == "on_demand"
    response = chat.invoke(
        [message],
    )
    assert response.response_metadata.get("service_tier") == "on_demand"

    # If an `invoke` call is made with no service tier, we fall back to the class level
    # setting
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="auto")
    response = chat.invoke(
        [message],
    )
    assert response.response_metadata.get("service_tier") == "auto"

    response = chat.invoke(
        [message],
        service_tier="on_demand",
    )
    assert response.response_metadata.get("service_tier") == "on_demand"

    with pytest.raises(BadRequestError):
        response = chat.invoke(
            [message],
            service_tier="invalid",
        )

    response = chat.invoke(
        [message],
        service_tier=None,
    )
    assert response.response_metadata.get("service_tier") == "auto"


def test_setting_service_tier_streaming() -> None:
    """Test service tier settings for streaming calls."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="flex")
    chunks = list(chat.stream("Why is the sky blue?", service_tier="auto"))

    assert chunks[-1].response_metadata.get("service_tier") == "auto"


async def test_setting_service_tier_request_async() -> None:
    """Test async setting of service tier at the request level."""
    chat = ChatGroq(model=DEFAULT_MODEL_NAME, service_tier="flex")
    response = await chat.ainvoke("Hello!", service_tier="on_demand")

    assert response.response_metadata.get("service_tier") == "on_demand"


# Groq does not currently support N > 1
# @pytest.mark.scheduled
# def test_chat_multiple_completions() -> None:
#     """Test ChatGroq wrapper with multiple completions."""
#     chat = ChatGroq(max_tokens=10, n=5)
#     message = HumanMessage(content="Hello")
#     response = chat._generate([message])
#     assert isinstance(response, ChatResult)
#     assert len(response.generations) == 5
#     for generation in response.generations:
#          assert isinstance(generation.message, BaseMessage)
#          assert isinstance(generation.message.content, str)

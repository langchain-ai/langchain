"""Test AzureChatOpenAI wrapper."""

import json
import os
from typing import Any, Optional

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from pydantic import BaseModel

from langchain_openai import AzureChatOpenAI
from tests.unit_tests.fake.callbacks import FakeCallbackHandler

OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "")
OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE", "")
OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT_NAME = os.environ.get(
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", ""),
)


def _get_llm(**kwargs: Any) -> AzureChatOpenAI:
    return AzureChatOpenAI(  # type: ignore[call-arg, call-arg, call-arg]
        deployment_name=DEPLOYMENT_NAME,
        openai_api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
        **kwargs,
    )


@pytest.mark.scheduled
@pytest.fixture
def llm() -> AzureChatOpenAI:
    return _get_llm(max_tokens=50)


def test_chat_openai(llm: AzureChatOpenAI) -> None:
    """Test AzureChatOpenAI wrapper."""
    message = HumanMessage(content="Hello")
    response = llm.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_openai_generate() -> None:
    """Test AzureChatOpenAI wrapper with generate."""
    chat = _get_llm(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
def test_chat_openai_multiple_completions() -> None:
    """Test AzureChatOpenAI wrapper with multiple completions."""
    chat = _get_llm(max_tokens=10, n=5)
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
    chat = _get_llm(
        max_tokens=10,
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
    chat = _get_llm(max_tokens=2, temperature=0, callback_manager=callback_manager)
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello!"


@pytest.mark.scheduled
async def test_async_chat_openai() -> None:
    """Test async generation."""
    chat = _get_llm(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
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
    chat = _get_llm(
        max_tokens=10,
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
def test_openai_streaming(llm: AzureChatOpenAI) -> None:
    """Test streaming tokens from OpenAI."""
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("I'm Pickle Rick"):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata.get("model_name") is not None


@pytest.mark.scheduled
async def test_openai_astream(llm: AzureChatOpenAI) -> None:
    """Test streaming tokens from OpenAI."""

    full: Optional[BaseMessageChunk] = None
    async for chunk in llm.astream("I'm Pickle Rick"):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert full.response_metadata.get("model_name") is not None


@pytest.mark.scheduled
async def test_openai_abatch(llm: AzureChatOpenAI) -> None:
    """Test streaming tokens from AzureChatOpenAI."""

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_abatch_tags(llm: AzureChatOpenAI) -> None:
    """Test batch tokens from AzureChatOpenAI."""

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_openai_batch(llm: AzureChatOpenAI) -> None:
    """Test batch tokens from AzureChatOpenAI."""

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_openai_ainvoke(llm: AzureChatOpenAI) -> None:
    """Test invoke tokens from AzureChatOpenAI."""

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)
    assert result.response_metadata.get("model_name") is not None


@pytest.mark.scheduled
def test_openai_invoke(llm: AzureChatOpenAI) -> None:
    """Test invoke tokens from AzureChatOpenAI."""

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
    assert result.response_metadata.get("model_name") is not None


def test_json_mode(llm: AzureChatOpenAI) -> None:
    response = llm.invoke(
        "Return this as json: {'a': 1}", response_format={"type": "json_object"}
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


async def test_json_mode_async(llm: AzureChatOpenAI) -> None:
    response = await llm.ainvoke(
        "Return this as json: {'a': 1}", response_format={"type": "json_object"}
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


class Foo(BaseModel):
    response: str


def test_stream_response_format(llm: AzureChatOpenAI) -> None:
    full: Optional[BaseMessageChunk] = None
    chunks = []
    for chunk in llm.stream("how are ya", response_format=Foo):
        chunks.append(chunk)
        full = chunk if full is None else full + chunk
    assert len(chunks) > 1
    assert isinstance(full, AIMessageChunk)
    parsed = full.additional_kwargs["parsed"]
    assert isinstance(parsed, Foo)
    assert isinstance(full.content, str)
    parsed_content = json.loads(full.content)
    assert parsed.response == parsed_content["response"]


async def test_astream_response_format(llm: AzureChatOpenAI) -> None:
    full: Optional[BaseMessageChunk] = None
    chunks = []
    async for chunk in llm.astream("how are ya", response_format=Foo):
        chunks.append(chunk)
        full = chunk if full is None else full + chunk
    assert len(chunks) > 1
    assert isinstance(full, AIMessageChunk)
    parsed = full.additional_kwargs["parsed"]
    assert isinstance(parsed, Foo)
    assert isinstance(full.content, str)
    parsed_content = json.loads(full.content)
    assert parsed.response == parsed_content["response"]

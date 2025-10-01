"""Test ChatBaseten chat model integration."""

import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_baseten import ChatBaseten


@pytest.mark.requires("baseten_api_key")
def test_chat_baseten_invoke() -> None:
    """Test ChatBaseten invoke."""
    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")

    chat = ChatBaseten(
        model="deepseek-ai/DeepSeek-V3-0324",
        baseten_api_key=api_key,
        temperature=0,
        max_tokens=50,
    )

    message = HumanMessage(content="Hello, how are you?")
    response = chat.invoke([message])

    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.requires("baseten_api_key")
def test_chat_baseten_stream() -> None:
    """Test ChatBaseten streaming."""
    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")

    chat = ChatBaseten(
        model="deepseek-ai/DeepSeek-V3-0324",
        baseten_api_key=api_key,
        temperature=0,
        max_tokens=50,
        streaming=True,
    )

    message = HumanMessage(content="Count from 1 to 5")
    chunks = list(chat.stream([message]))

    assert len(chunks) > 0
    content = "".join(chunk.content for chunk in chunks)
    assert len(content) > 0


@pytest.mark.requires("baseten_api_key")
async def test_chat_baseten_ainvoke() -> None:
    """Test ChatBaseten async invoke."""
    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")

    chat = ChatBaseten(
        model="deepseek-ai/DeepSeek-V3-0324",
        baseten_api_key=api_key,
        temperature=0,
        max_tokens=50,
    )

    message = HumanMessage(content="Hello, how are you?")
    response = await chat.ainvoke([message])

    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.requires("baseten_api_key")
async def test_chat_baseten_astream() -> None:
    """Test ChatBaseten async streaming."""
    api_key = os.environ.get("BASETEN_API_KEY")
    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")

    chat = ChatBaseten(
        model="deepseek-ai/DeepSeek-V3-0324",
        baseten_api_key=api_key,
        temperature=0,
        max_tokens=50,
        streaming=True,
    )

    message = HumanMessage(content="Count from 1 to 5")
    chunks = []
    async for chunk in chat.astream([message]):
        chunks.append(chunk)

    assert len(chunks) > 0
    content = "".join(chunk.content for chunk in chunks)
    assert len(content) > 0


@pytest.mark.requires("baseten_api_key")
@pytest.mark.requires("baseten_dedicated_model_url")
def test_chat_baseten_dedicated_url_invoke() -> None:
    """Test ChatBaseten with dedicated model URL."""
    api_key = os.environ.get("BASETEN_API_KEY")
    model_url = os.environ.get("BASETEN_DEDICATED_MODEL_URL")

    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")
    if not model_url:
        pytest.skip("BASETEN_DEDICATED_MODEL_URL not set")

    chat = ChatBaseten(
        model="dedicated-model",
        model_url=model_url,
        baseten_api_key=api_key,
        temperature=0,
        max_tokens=50,
    )

    message = HumanMessage(content="Hello from dedicated model!")
    response = chat.invoke([message])

    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.requires("baseten_api_key")
@pytest.mark.requires("baseten_dedicated_model_url")
def test_chat_baseten_dedicated_url_stream() -> None:
    """Test ChatBaseten dedicated URL streaming."""
    api_key = os.environ.get("BASETEN_API_KEY")
    model_url = os.environ.get("BASETEN_DEDICATED_MODEL_URL")

    if not api_key:
        pytest.skip("BASETEN_API_KEY not set")
    if not model_url:
        pytest.skip("BASETEN_DEDICATED_MODEL_URL not set")

    chat = ChatBaseten(
        model="dedicated-model",
        model_url=model_url,
        baseten_api_key=api_key,
        temperature=0,
        max_tokens=30,
        streaming=True,
    )

    message = HumanMessage(content="Count to 3")
    chunks = list(chat.stream([message]))

    assert len(chunks) > 0
    content = "".join(chunk.content for chunk in chunks)
    assert len(content) > 0

"""Test ChatStraico wrapper."""

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import (
    ChatGeneration,
    LLMResult,
)

from langchain_community.chat_models import ChatStraico


@pytest.mark.scheduled
def test_chat_straico() -> None:
    """Test ChatStraico wrapper."""
    chat = ChatStraico()
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_straico_model() -> None:
    """Test ChatStraico wrapper handles model_name."""
    chat = ChatStraico(model="foo")
    assert chat.model == "foo"
    chat = ChatStraico(model="bar")
    assert chat.model == "bar"


def test_chat_straico_system_message() -> None:
    """Test ChatStraico wrapper with system message."""
    chat = ChatStraico(model="google/gemini-pro")
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_straico_generate() -> None:
    """Test ChatStraico wrapper with generate."""
    chat = ChatStraico(model="google/gemini-pro")
    message = HumanMessage(content="Hello")
    message2 = HumanMessage(content="Hello2")
    response = chat.generate([[message], [message2]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_chat_straico_llm_output_contains_model_name() -> None:
    """Test llm_output contains model_name."""
    chat = ChatStraico()
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model


@pytest.mark.asyncio
async def test_async_chat_straico() -> None:
    """Test async generation."""
    chat = ChatStraico(model="google/gemini-pro")
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    assert response.llm_output
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.asyncio
async def test_openai_abatch_tags() -> None:
    """Test batch tokens from ChatStraico."""
    llm = ChatStraico()

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_openai_batch() -> None:
    """Test batch tokens from ChatStraico."""
    llm = ChatStraico()
    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.asyncio
async def test_openai_ainvoke() -> None:
    """Test invoke tokens from ChatStraico."""
    llm = ChatStraico()

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_openai_invoke() -> None:
    """Test invoke tokens from ChatStraico."""
    llm = ChatStraico()

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)

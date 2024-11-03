"""Test ChatPremAI from PremAI API wrapper.

Note: This test must be run with the PREMAI_API_KEY environment variable set to a valid
API key and a valid project_id.
For this we need to have a project setup in PremAI's platform: https://app.premai.io
"""

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models import ChatPremAI


@pytest.fixture
def chat() -> ChatPremAI:
    return ChatPremAI(project_id=8)  # type: ignore[call-arg]


def test_chat_premai() -> None:
    """Test ChatPremAI wrapper."""
    chat = ChatPremAI(project_id=8)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_prem_system_message() -> None:
    """Test ChatPremAI wrapper for system message"""
    chat = ChatPremAI(project_id=8)  # type: ignore[call-arg]
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_prem_model() -> None:
    """Test ChatPremAI wrapper handles model_name."""
    chat = ChatPremAI(model="foo", project_id=8)  # type: ignore[call-arg]
    assert chat.model == "foo"


def test_chat_prem_generate() -> None:
    """Test ChatPremAI wrapper with generate."""
    chat = ChatPremAI(project_id=8)  # type: ignore[call-arg]
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


async def test_prem_invoke(chat: ChatPremAI) -> None:
    """Tests chat completion with invoke"""
    result = chat.invoke("How is the weather in New York today?")
    assert isinstance(result.content, str)


def test_prem_streaming() -> None:
    """Test streaming tokens from Prem."""
    chat = ChatPremAI(project_id=8, streaming=True)  # type: ignore[call-arg]

    for token in chat.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)

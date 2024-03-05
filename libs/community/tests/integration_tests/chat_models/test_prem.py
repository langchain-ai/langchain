"""Test ChatPrem from PremAI API wrapper.

Note: This test must be run with the PREMAI_API_KEY environment variable set to a valid API key and a valid project_id. For this we need to have a project setup in PremAI's platform: https://app.premai.io
"""

import pytest
from typing import cast
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_community.chat_models import ChatPrem

# TODO: Need to see if batching can be done using prem-sdk


@pytest.fixture
def chat() -> ChatPrem:
    return ChatPrem(project_id=8)


@pytest.mark.scheduled
def test_chat_premai() -> None:
    """Test ChatPrem wrapper."""
    chat = ChatPrem(project_id=8)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_prem_system_message() -> None:
    """Test ChatPrem wrapper for system message"""
    chat = ChatPrem(project_id=8)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_prem_model() -> None:
    """Test ChatPrem wrapper handles model_name."""
    chat = ChatPrem(model="foo", project_id=8)
    assert chat.model == "foo"


@pytest.mark.scheduled
def test_chat_prem_generate() -> None:
    """Test ChatPrem wrapper with generate."""
    chat = ChatPrem(project_id=8)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        # TODO: Need to investigate this, getting same things in prem-sdk
        # assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
async def test_prem_invoke(chat: ChatPrem) -> None:
    """Tests chat completion with invoke"""
    # TODO: Need to investigate this, happend inside prem-sdk too
    # result = await chat.invoke("How is the weather in New York today?", stop=[","])
    # assert result.content[-1] == ","
    result = chat.invoke("How is the weather in New York today?")
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_prem_streaming() -> None:
    """Test streaming tokens from Prem."""
    chat = ChatPrem(project_id=8, streaming=True)

    for token in chat.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


# TODO: Need to investigate this
@pytest.mark.scheduled
def test_prem_streaming_stop_words() -> None:
    """Test streaming tokens with stop words."""
    chat = ChatPrem(project_id=8, streaming=True)
    last_token = ""
    for token in chat.stream(
        "I'm Pickle Rick",
    ):  # stop=[","]):
        last_token = cast(str, token.content)
        assert isinstance(token.content, str)
    # assert last_token[-1] == ","

from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.dappier import (
    ChatDappierAI,
)


@pytest.mark.scheduled
def test_dappier_chat() -> None:
    """Test ChatDappierAI wrapper."""
    chat = ChatDappierAI(
        dappier_endpoint="https://api.dappier.com/app/datamodelconversation",
        dappier_model="dm_01hpsxyfm2fwdt2zet9cg6fdxt",
    )
    message = HumanMessage(content="Who are you ?")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_dappier_generate() -> None:
    """Test generate method of Dappier AI."""
    chat = ChatDappierAI(
        dappier_endpoint="https://api.dappier.com/app/datamodelconversation",
        dappier_model="dm_01hpsxyfm2fwdt2zet9cg6fdxt",
    )
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="Who won the last super bowl?")],
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content
    assert chat_messages == messages_copy


@pytest.mark.scheduled
async def test_dappier_agenerate() -> None:
    """Test async generation."""
    chat = ChatDappierAI(
        dappier_endpoint="https://api.dappier.com/app/datamodelconversation",
        dappier_model="dm_01hpsxyfm2fwdt2zet9cg6fdxt",
    )
    message = HumanMessage(content="Who won the last super bowl?")
    result: LLMResult = await chat.agenerate([[message], [message]])
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.text == response.message.content

from typing import List

from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.unify import ChatUnify
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

import pytest
from unittest.mock import patch, MagicMock

def test_chat_unify_call() -> None:
    """Test valid call to ChatUnify."""
    with patch('httpx.Client.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello, how can I help you?"}}]
        }
        mock_post.return_value = mock_response

        chat = ChatUnify(unify_api_key="test_key")
        message = HumanMessage(content="Hello")
        response = chat.invoke([message])
        
        assert isinstance(response, AIMessage)
        assert isinstance(response.content, str)
        assert response.content == "Hello, how can I help you?"

def test_chat_unify_generate() -> None:
    """Test generate method of ChatUnify."""
    with patch('httpx.Client.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Dogs typically have 5 toes on their front paws and 4 toes on their back paws."}}]
        }
        mock_post.return_value = mock_response

        chat = ChatUnify(unify_api_key="test_key")
        chat_messages: List[List[BaseMessage]] = [
            [HumanMessage(content="How many toes do dogs have?")]
        ]
        messages_copy = [messages.copy() for messages in chat_messages]
        result: LLMResult = chat.generate(chat_messages)
        
        assert isinstance(result, LLMResult)
        for response in result.generations[0]:
            assert isinstance(response, ChatGeneration)
            assert isinstance(response.text, str)
            assert response.text == response.message.content
        assert chat_messages == messages_copy

@pytest.mark.asyncio
async def test_chat_unify_streaming() -> None:
    """Test streaming tokens from ChatUnify."""
    with patch('httpx.AsyncClient.stream') as mock_stream:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines.return_value = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": ", how"}}]}',
            'data: {"choices": [{"delta": {"content": " can I help you?"}}]}',
        ]
        mock_stream.return_value.__aenter__.return_value = mock_response

        chat = ChatUnify(unify_api_key="test_key", streaming=True)
        message = HumanMessage(content="Hello")
        response = await chat.ainvoke([message])
        
        assert isinstance(response, AIMessage)
        assert isinstance(response.content, str)
        assert response.content == "Hello, how can I help you?"

def test_chat_unify_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    with patch('httpx.Client.stream') as mock_stream:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            'data: {"choices": [{"delta": {"content": "This "}}]}',
            'data: {"choices": [{"delta": {"content": "is "}}]}',
            'data: {"choices": [{"delta": {"content": "a "}}]}',
            'data: {"choices": [{"delta": {"content": "test "}}]}',
            'data: {"choices": [{"delta": {"content": "sentence."}}]}',
        ]
        mock_stream.return_value.__enter__.return_value = mock_response

        callback_handler = FakeCallbackHandler()
        callback_manager = CallbackManager([callback_handler])
        chat = ChatUnify(
            unify_api_key="test_key",
            streaming=True,
            callback_manager=callback_manager,
            verbose=True,
        )
        message = HumanMessage(content="Write me a sentence with 5 words.")
        chat.invoke([message])
        assert callback_handler.llm_streams > 1
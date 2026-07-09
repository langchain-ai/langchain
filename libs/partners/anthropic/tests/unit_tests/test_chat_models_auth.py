import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

def test_chat_anthropic_callable_api_key() -> None:
    mock_key = "mock-key-from-callable"
    def get_key() -> str:
        return mock_key
    
    llm = ChatAnthropic(api_key=get_key, model="claude-3-opus-20240229")
    
    with patch.object(llm._client.messages, "create") as mock_create:
        mock_message = MagicMock()
        mock_message.text = "response"
        mock_message.type = "text"
        mock_response = MagicMock(content=[mock_message])
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=10)
        mock_response.id = "msg_123"
        mock_response.model = "claude-3-opus-20240229"
        mock_response.role = "assistant"
        mock_create.return_value = mock_response
        
        llm.invoke("Hello")
        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert "extra_headers" in kwargs
        assert kwargs["extra_headers"]["x-api-key"] == mock_key

def test_chat_anthropic_auth_token_provider() -> None:
    mock_token = "mock-token"
    def get_token() -> str:
        return mock_token
    
    llm = ChatAnthropic(auth_token_provider=get_token, model="claude-3-opus-20240229")
    
    with patch.object(llm._client.messages, "create") as mock_create:
        mock_message = MagicMock()
        mock_message.text = "response"
        mock_message.type = "text"
        mock_response = MagicMock(content=[mock_message])
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=10)
        mock_response.id = "msg_123"
        mock_response.model = "claude-3-opus-20240229"
        mock_response.role = "assistant"
        mock_create.return_value = mock_response

        llm.invoke("Hello")
        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert "extra_headers" in kwargs
        assert kwargs["extra_headers"]["Authorization"] == f"Bearer {mock_token}"

@pytest.mark.asyncio
async def test_chat_anthropic_async_auth_token_provider() -> None:
    mock_token = "mock-async-token"
    async def get_token() -> str:
        await asyncio.sleep(0.01)
        return mock_token
    
    llm = ChatAnthropic(auth_token_provider=get_token, model="claude-3-opus-20240229")
    
    with patch.object(llm._async_client.messages, "create") as mock_create:
        mock_message = MagicMock()
        mock_message.text = "response"
        mock_message.type = "text"
        mock_response = MagicMock(content=[mock_message])
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=10)
        mock_response.id = "msg_123"
        mock_response.model = "claude-3-opus-20240229"
        mock_response.role = "assistant"
        mock_create.return_value = mock_response

        await llm.ainvoke("Hello")
        mock_create.assert_called_once()
        kwargs = mock_create.call_args.kwargs
        assert "extra_headers" in kwargs
        assert kwargs["extra_headers"]["Authorization"] == f"Bearer {mock_token}"

#!/usr/bin/env python3
"""
Test for the fix that handles Ollama 'load' responses with empty content.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


def test_load_response_with_empty_content_is_skipped():
    """Test that load responses with empty content are skipped."""
    load_only_response = [
        {
            'model': 'test-model',
            'created_at': '2025-01-01T00:00:00.000000000Z',
            'done': True,
            'done_reason': 'load',
            'message': {
                'role': 'assistant',
                'content': ''
            }
        }
    ]
    
    with patch('langchain_ollama.chat_models.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_only_response
        
        llm = ChatOllama(model='test-model')
        
        with pytest.raises(ValueError, match="No data received from Ollama stream"):
            llm.invoke([HumanMessage('Hello')])


def test_load_response_with_whitespace_content_is_skipped():
    """Test that load responses with only whitespace content are skipped."""
    load_whitespace_response = [
        {
            'model': 'test-model',
            'created_at': '2025-01-01T00:00:00.000000000Z',
            'done': True,
            'done_reason': 'load',
            'message': {
                'role': 'assistant',
                'content': '   \n  \t  '
            }
        }
    ]
    
    with patch('langchain_ollama.chat_models.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_whitespace_response
        
        llm = ChatOllama(model='test-model')
        
        with pytest.raises(ValueError, match="No data received from Ollama stream"):
            llm.invoke([HumanMessage('Hello')])


def test_load_response_with_actual_content_is_not_skipped():
    """Test that load responses with actual content are NOT skipped."""
    load_with_content_response = [
        {
            'model': 'test-model',
            'created_at': '2025-01-01T00:00:00.000000000Z',
            'done': True,
            'done_reason': 'load',
            'message': {
                'role': 'assistant',
                'content': 'This is actual content'
            }
        }
    ]
    
    with patch('langchain_ollama.chat_models.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_with_content_response
        
        llm = ChatOllama(model='test-model')
        result = llm.invoke([HumanMessage('Hello')])
        
        assert result.content == 'This is actual content'
        assert result.response_metadata.get('done_reason') == 'load'


def test_load_followed_by_content_response():
    """Test that load responses are skipped when followed by actual content."""
    load_then_content_response = [
        {
            'model': 'test-model',
            'created_at': '2025-01-01T00:00:00.000000000Z',
            'done': True,
            'done_reason': 'load',
            'message': {
                'role': 'assistant',
                'content': ''
            }
        },
        {
            'model': 'test-model',
            'created_at': '2025-01-01T00:00:01.000000000Z',
            'done': True,
            'done_reason': 'stop',
            'message': {
                'role': 'assistant',
                'content': 'Hello! How can I help you today?'
            }
        }
    ]
    
    with patch('langchain_ollama.chat_models.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = load_then_content_response
        
        llm = ChatOllama(model='test-model')
        result = llm.invoke([HumanMessage('Hello')])
        
        assert result.content == 'Hello! How can I help you today?'
        assert result.response_metadata.get('done_reason') == 'stop'


def test_normal_stop_response_unaffected():
    """Test that normal stop responses continue to work as before."""
    normal_response = [
        {
            'model': 'test-model',
            'created_at': '2025-01-01T00:00:00.000000000Z',
            'done': True,
            'done_reason': 'stop',
            'message': {
                'role': 'assistant',
                'content': 'Normal response'
            }
        }
    ]
    
    with patch('langchain_ollama.chat_models.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = normal_response
        
        llm = ChatOllama(model='test-model')
        result = llm.invoke([HumanMessage('Hello')])
        
        assert result.content == 'Normal response'
        assert result.response_metadata.get('done_reason') == 'stop'


@pytest.mark.asyncio
async def test_async_load_response_with_empty_content_is_skipped():
    """Test that load responses with empty content are skipped in async mode."""
    load_only_response = [
        {
            'model': 'test-model',
            'created_at': '2025-01-01T00:00:00.000000000Z',
            'done': True,
            'done_reason': 'load',
            'message': {
                'role': 'assistant',
                'content': ''
            }
        }
    ]
    
    with patch('langchain_ollama.chat_models.AsyncClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        async def async_chat_generator():
            for response in load_only_response:
                yield response
        
        mock_client.chat.return_value = async_chat_generator()
        
        llm = ChatOllama(model='test-model')
        
        with pytest.raises(ValueError, match="No data received from Ollama stream"):
            await llm.ainvoke([HumanMessage('Hello')])


if __name__ == "__main__":
    # Run the tests
    test_load_response_with_empty_content_is_skipped()
    print("✓ test_load_response_with_empty_content_is_skipped passed")
    
    test_load_response_with_whitespace_content_is_skipped() 
    print("✓ test_load_response_with_whitespace_content_is_skipped passed")
    
    test_load_response_with_actual_content_is_not_skipped()
    print("✓ test_load_response_with_actual_content_is_not_skipped passed")
    
    test_load_followed_by_content_response()
    print("✓ test_load_followed_by_content_response passed")
    
    test_normal_stop_response_unaffected()
    print("✓ test_normal_stop_response_unaffected passed")
    
    print("\nAll tests passed! 🎉")
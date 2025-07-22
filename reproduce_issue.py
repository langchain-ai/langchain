#!/usr/bin/env python3
"""
Script to reproduce the issue where ChatOllama returns empty content with done_reason: 'load'
"""

from unittest.mock import MagicMock, patch
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.config import RunnableConfig
import json
from httpx import Response, Request


def mock_ollama_response_with_load():
    """Mock an Ollama response that has done_reason: 'load' and empty content"""
    return {
        'model': 'qwen2.5:7b',
        'created_at': '2025-05-21T09:37:56.153539017Z',
        'done': True,
        'done_reason': 'load',
        'message': {
            'role': 'assistant',
            'content': ''
        },
        'total_duration': None,
        'load_duration': None,
        'prompt_eval_count': None,
        'prompt_eval_duration': None,
        'eval_count': None,
        'eval_duration': None,
    }

def mock_ollama_response_with_content():
    """Mock an Ollama response with actual content"""
    return {
        'model': 'qwen2.5:7b',
        'created_at': '2025-05-21T09:37:56.153539017Z',
        'done': True,
        'done_reason': 'stop',
        'message': {
            'role': 'assistant',
            'content': 'Hello! How can I help you today?'
        },
        'total_duration': 1234567890,
        'load_duration': 123456789,
        'prompt_eval_count': 10,
        'prompt_eval_duration': 50000000,
        'eval_count': 20,
        'eval_duration': 100000000,
    }


def test_reproduce_issue():
    """Reproduce the issue with empty content and done_reason: 'load'"""
    
    # Mock the Ollama client to return a load response
    with patch('langchain_ollama.chat_models.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the chat method to return the problematic response
        mock_client.chat.return_value = [mock_ollama_response_with_load()]
        
        llm = ChatOllama(model='qwen2.5:7b', temperature=0)
        tools = []
        agent = create_react_agent(
            model=llm,
            tools=tools
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "1"
            }
        }
        
        result = agent.invoke(HumanMessage('Hello'), config=config)
        
        print("=== PROBLEMATIC RESULT ===")
        print(f"Content: '{result['messages'][-1].content}'")
        print(f"Response metadata: {result['messages'][-1].response_metadata}")
        
        # Verify this is the issue
        assert result["messages"][-1].content == "", "Content should be empty"
        assert result["messages"][-1].response_metadata.get('done_reason') == 'load', "Should have done_reason: 'load'"
        

def test_expected_behavior():
    """Test what the expected behavior should be"""
    
    # Mock the Ollama client to return a proper response
    with patch('langchain_ollama.chat_models.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the chat method to return a proper response
        mock_client.chat.return_value = [mock_ollama_response_with_content()]
        
        llm = ChatOllama(model='qwen2.5:7b', temperature=0)
        tools = []
        agent = create_react_agent(
            model=llm,
            tools=tools
        )
        config: RunnableConfig = {
            "configurable": {
                "thread_id": "1"
            }
        }
        
        result = agent.invoke(HumanMessage('Hello'), config=config)
        
        print("\n=== EXPECTED RESULT ===")
        print(f"Content: '{result['messages'][-1].content}'")
        print(f"Response metadata: {result['messages'][-1].response_metadata}")
        
        # Verify this is the expected behavior
        assert result["messages"][-1].content != "", "Content should not be empty"
        assert result["messages"][-1].response_metadata.get('done_reason') == 'stop', "Should have done_reason: 'stop'"


def test_direct_chat_ollama():
    """Test ChatOllama directly"""
    
    print("\n=== TESTING DIRECT CHATOLLAMA ===")
    
    # Test with load response
    with patch('langchain_ollama.chat_models.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the chat method to return the problematic response
        mock_client.chat.return_value = [mock_ollama_response_with_load()]
        
        llm = ChatOllama(model='qwen2.5:7b', temperature=0)
        result = llm.invoke(HumanMessage('Hello'))
        
        print(f"Load response - Content: '{result.content}'")
        print(f"Load response - Metadata: {result.response_metadata}")
        
    # Test with proper response
    with patch('langchain_ollama.chat_models.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the chat method to return proper response
        mock_client.chat.return_value = [mock_ollama_response_with_content()]
        
        llm = ChatOllama(model='qwen2.5:7b', temperature=0)
        result = llm.invoke(HumanMessage('Hello'))
        
        print(f"Stop response - Content: '{result.content}'")
        print(f"Stop response - Metadata: {result.response_metadata}")


if __name__ == "__main__":
    try:
        print("Reproducing the issue...")
        test_reproduce_issue()
        print("✓ Issue reproduced successfully")
        
        print("\nTesting expected behavior...")
        test_expected_behavior()
        print("✓ Expected behavior confirmed")
        
        print("\nTesting direct ChatOllama...")
        test_direct_chat_ollama()
        print("✓ Direct test completed")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
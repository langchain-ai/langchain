"""Test case for vLLM compatibility issue #32252"""

from unittest.mock import MagicMock

import pytest

from langchain_openai.chat_models.base import BaseChatOpenAI


def create_mock_chat_openai():
    """Create a mock ChatOpenAI instance for testing _create_chat_result"""
    # Create a mock instance that has the _create_chat_result method
    mock_instance = MagicMock()
    # Bind the actual method to the mock
    mock_instance._create_chat_result = BaseChatOpenAI._create_chat_result.__get__(
        mock_instance, BaseChatOpenAI
    )
    return mock_instance


def test_vllm_response_parsing_fix():
    """Test that our fix handles the vLLM compatibility issue correctly"""

    # Mock vLLM response data
    mock_vllm_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "This is a test response from vLLM",
                    "reasoning_content": None,
                    "role": "assistant",
                    "tool_calls": [],
                },
                "stop_reason": None,
            }
        ],
        "created": 1753518740,
        "id": "chatcmpl-test-id",
        "model": "test-model",
        "object": "chat.completion",
        "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
    }

    # Create a mock response object that simulates the vLLM/OpenAI client issue
    mock_response_obj = MagicMock()

    # Simulate the problematic behavior: model_dump() returns choices as None
    broken_response_dict = mock_vllm_response.copy()
    broken_response_dict["choices"] = None
    mock_response_obj.model_dump.return_value = broken_response_dict

    # But the response object itself has the choices attribute with valid data
    mock_choice = MagicMock()
    mock_choice.model_dump.return_value = mock_vllm_response["choices"][0]
    mock_response_obj.choices = [mock_choice]

    llm = create_mock_chat_openai()

    # With our fix, this should work correctly by accessing response.choices directly
    result = llm._create_chat_result(mock_response_obj)

    assert result is not None
    assert len(result.generations) == 1
    assert result.generations[0].message.content == "This is a test response from vLLM"


def test_vllm_response_parsing():
    """Test that reproduces the original vLLM compatibility issue (now fixed)"""

    # Mock vLLM response that was working with direct requests/OpenAI client
    mock_vllm_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "This is a test response from vLLM",
                    "reasoning_content": None,
                    "role": "assistant",
                    "tool_calls": [],
                },
                "stop_reason": None,
            }
        ],
        "created": 1753518740,
        "id": "chatcmpl-test-id",
        "kv_transfer_params": None,
        "model": "test-model",
        "object": "chat.completion",
        "prompt_logprobs": None,
        "usage": {
            "completion_tokens": 10,
            "prompt_tokens": 20,
            "prompt_tokens_details": None,
            "total_tokens": 30,
        },
    }

    # Create a mock response object that simulates the problematic
    # OpenAI client behavior where model_dump() returns choices as None
    # even though the original response has valid choices
    mock_response_obj = MagicMock()
    # This simulates the bug: model_dump() returns choices as None despite
    # valid data
    broken_response_dict = mock_vllm_response.copy()
    broken_response_dict["choices"] = (
        None  # This is the bug - OpenAI client parsing issue
    )
    mock_response_obj.model_dump.return_value = broken_response_dict

    # But the original response object doesn't have choices attribute
    # (simulating the worst case)
    mock_response_obj.choices = None

    llm = create_mock_chat_openai()

    # This should still raise the TypeError because we can't recover choices
    # from anywhere
    with pytest.raises(
        TypeError, match="Received response with null value for `choices`"
    ):
        llm._create_chat_result(mock_response_obj)


def test_vllm_response_parsing_working_case():
    """Test a working case where the response object behaves correctly"""

    # Mock vLLM response
    mock_vllm_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "This is a test response from vLLM",
                    "reasoning_content": None,
                    "role": "assistant",
                    "tool_calls": [],
                },
                "stop_reason": None,
            }
        ],
        "created": 1753518740,
        "id": "chatcmpl-test-id",
        "model": "test-model",
        "object": "chat.completion",
        "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
    }

    # Test with a working response object where model_dump() returns correct data
    mock_response_obj = MagicMock()
    mock_response_obj.model_dump.return_value = mock_vllm_response

    llm = create_mock_chat_openai()

    # This should work correctly
    result = llm._create_chat_result(mock_response_obj)

    assert len(result.generations) == 1
    assert result.generations[0].message.content == "This is a test response from vLLM"


def test_vllm_response_parsing_with_dict():
    """Test that vLLM responses work when passed as dict"""

    mock_vllm_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "This is a test response from vLLM",
                    "reasoning_content": None,
                    "role": "assistant",
                    "tool_calls": [],
                },
                "stop_reason": None,
            }
        ],
        "created": 1753518740,
        "id": "chatcmpl-test-id",
        "model": "test-model",
        "object": "chat.completion",
        "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
    }

    llm = create_mock_chat_openai()

    # This should work fine since it's a dict
    result = llm._create_chat_result(mock_vllm_response)

    assert len(result.generations) == 1
    assert result.generations[0].message.content == "This is a test response from vLLM"


def test_vllm_edge_case_null_choices():
    """Test edge case where choices is actually null"""

    mock_vllm_response = {
        "choices": None,  # This should raise the error
        "created": 1753518740,
        "id": "chatcmpl-test-id",
        "model": "test-model",
        "object": "chat.completion",
    }

    llm = create_mock_chat_openai()

    with pytest.raises(
        TypeError, match="Received response with null value for `choices`"
    ):
        llm._create_chat_result(mock_vllm_response)


def test_vllm_edge_case_missing_choices():
    """Test edge case where choices key is missing"""

    mock_vllm_response = {
        "created": 1753518740,
        "id": "chatcmpl-test-id",
        "model": "test-model",
        "object": "chat.completion",
    }

    llm = create_mock_chat_openai()

    with pytest.raises(KeyError, match="Response missing `choices` key"):
        llm._create_chat_result(mock_vllm_response)

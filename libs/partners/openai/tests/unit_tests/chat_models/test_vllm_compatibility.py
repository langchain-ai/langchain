"""
Test for vLLM compatibility fix (issue #32252).

Tests the enhanced error handling when vLLM or other OpenAI-compatible APIs
return responses with choices=None.
"""

import pytest
from pydantic import SecretStr

from langchain_openai.chat_models.base import ChatOpenAI


class TestVLLMCompatibility:
    """Test vLLM compatibility improvements."""

    def test_vllm_null_choices_error_message(self) -> None:
        """Test enhanced error message for vLLM responses with null choices."""
        llm = ChatOpenAI(api_key=SecretStr("test"), base_url="test")

        # Simulate a vLLM response with null choices and vLLM-specific fields
        vllm_response = {
            "choices": None,
            "created": 1753518740,
            "id": "chatcmpl-test",
            "kv_transfer_params": None,  # vLLM-specific field
            "model": "test-model",
            "object": "chat.completion",
            "prompt_logprobs": None,  # vLLM-specific field
            "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
        }

        with pytest.raises(TypeError) as exc_info:
            llm._create_chat_result(vllm_response)

        error_msg = str(exc_info.value)

        # Verify enhanced error message contains helpful information
        assert "vLLM detected" in error_msg
        assert "Response ID: chatcmpl-test" in error_msg
        assert "Model: test-model" in error_msg
        assert "vLLM configuration issues" in error_msg
        assert "model loading problems" in error_msg
        assert "high server load" in error_msg
        assert "check vLLM logs" in error_msg

    def test_generic_api_null_choices_error_message(self) -> None:
        """Test enhanced error message for generic APIs with null choices."""
        llm = ChatOpenAI(api_key=SecretStr("test"), base_url="test")

        # Generic OpenAI-compatible API response with null choices
        generic_response = {
            "choices": None,
            "created": 123,
            "id": "test-id",
            "model": "generic-model",
            "object": "chat.completion",
            "usage": {"completion_tokens": 5, "prompt_tokens": 10, "total_tokens": 15},
        }

        with pytest.raises(TypeError) as exc_info:
            llm._create_chat_result(generic_response)

        error_msg = str(exc_info.value)

        # Should not detect vLLM for generic responses
        assert "vLLM detected" not in error_msg
        assert "Response ID: test-id" in error_msg
        assert "Model: generic-model" in error_msg
        assert "API endpoint" in error_msg

    def test_minimal_null_choices_error_message(self) -> None:
        """Test error message for minimal response with null choices."""
        llm = ChatOpenAI(api_key=SecretStr("test"), base_url="test")

        # Minimal response with just choices=None
        minimal_response = {"choices": None}

        with pytest.raises(TypeError) as exc_info:
            llm._create_chat_result(minimal_response)

        error_msg = str(exc_info.value)

        # Should include response keys for debugging
        assert "Response keys:" in error_msg

    def test_working_vllm_response(self) -> None:
        """Test that working vLLM responses are processed correctly."""
        llm = ChatOpenAI(api_key=SecretStr("test"), base_url="test")

        # Working vLLM response with vLLM-specific fields
        working_response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": "This is a test response from vLLM",
                        "reasoning_content": None,  # vLLM-specific
                        "role": "assistant",
                        "tool_calls": [],
                    },
                    "stop_reason": None,  # vLLM-specific
                }
            ],
            "created": 1753518740,
            "id": "chatcmpl-working",
            "kv_transfer_params": None,  # vLLM-specific
            "model": "test-model",
            "object": "chat.completion",
            "prompt_logprobs": None,  # vLLM-specific
            "usage": {
                "completion_tokens": 7,
                "prompt_tokens": 15,
                "prompt_tokens_details": None,  # vLLM-specific
                "total_tokens": 22,
            },
        }

        # Should process successfully without errors
        result = llm._create_chat_result(working_response)

        # Verify the result is properly structured
        assert len(result.generations) == 1
        assert result.generations[0].text == "This is a test response from vLLM"
        assert result.llm_output is not None
        assert result.llm_output["model_name"] == "test-model"
        assert result.llm_output["id"] == "chatcmpl-working"

"""Unit tests for ChatFireworks."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from langchain_fireworks.chat_models import _convert_dict_to_message


def test_convert_dict_to_message_with_reasoning_content() -> None:
    """Test that reasoning_content is correctly extracted from API response."""
    response_dict = {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Let me think about this step by step...",
    }

    message = _convert_dict_to_message(response_dict)

    assert isinstance(message, AIMessage)
    assert message.content == "The answer is 42."
    assert "reasoning_content" in message.additional_kwargs
    expected_reasoning = "Let me think about this step by step..."
    assert message.additional_kwargs["reasoning_content"] == expected_reasoning


def test_convert_dict_to_message_without_reasoning_content() -> None:
    """Test that messages without reasoning_content work correctly."""
    response_dict = {
        "role": "assistant",
        "content": "The answer is 42.",
    }

    message = _convert_dict_to_message(response_dict)

    assert isinstance(message, AIMessage)
    assert message.content == "The answer is 42."
    assert "reasoning_content" not in message.additional_kwargs


def test_ls_params_versions_value() -> None:
    """Test that _get_ls_params reports the correct langchain-fireworks version."""
    import os
    from importlib.metadata import version

    from langchain_fireworks.chat_models import ChatFireworks

    os.environ.setdefault("FIREWORKS_API_KEY", "fake-key")
    llm = ChatFireworks(model="accounts/fireworks/models/llama-v3-70b-instruct")
    ls_params = llm._get_ls_params()
    assert "versions" in ls_params
    assert ls_params["versions"] == {
        "langchain-fireworks": version("langchain-fireworks")
    }

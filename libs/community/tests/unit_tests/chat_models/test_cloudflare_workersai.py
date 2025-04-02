"""Test CloudflareWorkersAI Chat API wrapper."""

from langchain_core.messages import (
    HumanMessage,
)

from langchain_community.chat_models.cloudflare_workersai import (
    _convert_message_to_dict,
)


def test_convert_human_message() -> None:
    """Test converting a simple HumanMessage."""
    message = HumanMessage(content="Hello, AI!")
    result = _convert_message_to_dict(message)
    assert result == {"role": "user", "content": "Hello, AI!"}

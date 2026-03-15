"""Tests for strict parameter handling in Responses API"""

from langchain_core.messages import HumanMessage
from langchain_openai.chat_models.base import _construct_responses_api_payload


class TestStrictParameter:
    def test_strict_false_added_when_not_specified(self):
        """Test that strict=False is added when not specified in tool definition."""
        messages = [HumanMessage(content="test")]
        payload = {
            "model": "gpt-5",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
        }

        result = _construct_responses_api_payload(messages, payload)

        # strict should be added as False
        assert result["tools"][0]["strict"] is False

    def test_strict_preserved_when_explicitly_set(self):
        """Test that explicit strict value is preserved."""
        messages = [HumanMessage(content="test")]
        payload = {
            "model": "gpt-5",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                        "strict": True,
                    },
                }
            ],
        }

        result = _construct_responses_api_payload(messages, payload)

        # strict should be preserved as True
        assert result["tools"][0]["strict"] is True

    def test_strict_false_preserved_when_explicitly_set(self):
        """Test that explicit strict=False is preserved."""
        messages = [HumanMessage(content="test")]
        payload = {
            "model": "gpt-5",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                        "strict": False,
                    },
                }
            ],
        }

        result = _construct_responses_api_payload(messages, payload)

        # strict should be preserved as False
        assert result["tools"][0]["strict"] is False

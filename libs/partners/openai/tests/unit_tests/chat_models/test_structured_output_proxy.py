"""Test structured output compatibility with proxies."""

import json
from unittest.mock import MagicMock

from pydantic import BaseModel, Field, SecretStr

from langchain_openai import ChatOpenAI


class Joke(BaseModel):
    setup: str = Field(description="the setup of the joke")
    punchline: str = Field(description="the punchline")


def test_structured_output_proxy_fix() -> None:
    """Test that tool calls are respected even if finish_reason is 'stop'.

    This simulates behavior seen in some proxies (e.g. LiteLLM) where
    finish_reason is not correctly set to 'tool_calls'.
    """
    # Configure response as a DICT directly to avoid Pydantic/Mock model_dump issues
    # validation logic in base.py handles dicts natively
    mock_response_dict = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "Joke",
                                "arguments": json.dumps(
                                    {
                                        "setup": "Why did the chicken cross the road?",
                                        "punchline": "To get to the other side",
                                    }
                                ),
                            },
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 10},
        "model": "gpt-4",
    }

    # Initialize ChatOpenAI
    llm = ChatOpenAI(model="gpt-4", api_key=SecretStr("test"))

    # Inject Mock Client directly
    # _generate calls self.client.with_raw_response.create(...).parse()
    mock_raw_response = MagicMock()
    mock_raw_response.parse.return_value = mock_response_dict

    llm.client = MagicMock()
    llm.client.with_raw_response.create.return_value = mock_raw_response

    # Also mock root_client just in case logic paths change
    llm.root_client = MagicMock()
    llm.root_client.chat.completions.with_raw_response.parse.return_value = (
        mock_raw_response
    )

    # Use with_structured_output
    structured_llm = llm.with_structured_output(Joke)

    # Invoke
    result = structured_llm.invoke("Tell me a little joke")

    # Verification
    assert isinstance(result, Joke)
    assert result.punchline == "To get to the other side"

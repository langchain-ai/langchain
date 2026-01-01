import sys
from unittest.mock import MagicMock

# Mock dependencies before imports to handle missing packages
mock_anthropic = MagicMock()
mock_anthropic.__path__ = []
sys.modules["anthropic"] = mock_anthropic

mock_langsmith = MagicMock()
mock_langsmith.__path__ = []
sys.modules["langsmith"] = mock_langsmith

# Mock langchain_core if needed, but it should be available via libs/core
# But runnables might try to import langsmith.

import json
from unittest.mock import MagicMock, patch

from langchain.agents import create_agent
from langchain.tools import tool

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback mock if pydantic is missing
    BaseModel = MagicMock
    Field = MagicMock


# Define a simple tool that can be called from code_execution
class GetWeatherInput(BaseModel):
    """Input for weather lookup"""

    location: str = Field(description="City name")


@tool(
    args_schema=GetWeatherInput,
    extras={"allowed_callers": ["code_execution_20250825"]},
)
def get_weather(location: str) -> dict:
    """Get weather for a location. Returns temperature and conditions."""
    return {"location": location, "temp_f": 72, "conditions": "sunny"}


def reproduce():
    # Mock the Anthropic client to avoid actual API calls
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock()

    # Create ChatAnthropic with mocked client
    model = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        api_key="sk-ant-fake-key",
        betas=["advanced-tool-use-2025-11-20"],
    )
    # Patch the _client property to return our mock
    with patch.object(ChatAnthropic, "_client", return_value=mock_client):
        agent = create_agent(
            model=model,
            tools=[
                {"type": "code_execution_20250825", "name": "code_execution"},
                get_weather,
            ],
            middleware=[AnthropicPromptCachingMiddleware()],
        )

        messages = [
            {
                "role": "user",
                "content": "Use code_execution to check the weather in NYC and SF, then summarize.",
            },
        ]

        from langchain_core.messages import AIMessage, HumanMessage

        # Simulating the conversation state
        test_messages = [
            HumanMessage(content="Use code_execution to check the weather in NYC."),
            AIMessage(
                content=[
                    {
                        "type": "tool_use",
                        "id": "toolu_013QtuxLfX1Dm5JN97F9ZkWL",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                        "caller": {
                            "type": "code_execution_20250825",
                            "tool_id": "srvtoolu_123",
                        },
                    }
                ]
            ),
        ]

        # Call _get_request_payload directly
        payload = model._get_request_payload(
            input_=test_messages, cache_control={"type": "ephemeral", "ttl": "5m"}
        )

        print("Debugging Payload Content blocks:")
        last_message = payload["messages"][-1]
        print(json.dumps(last_message, indent=2))

        content_blocks = last_message["content"]
        last_block = content_blocks[-1]

        if "cache_control" in last_block:
            print("\n[FAIL] cache_control found in the last block!")
            if "caller" in last_block:
                print(f"Block has caller: {last_block['caller']}")
            else:
                print("Block has no caller.")
        else:
            print("\n[PASS] cache_control NOT found in the last block.")


if __name__ == "__main__":
    try:
        reproduce()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()

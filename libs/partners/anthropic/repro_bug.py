import sys
from unittest.mock import MagicMock


# Simple mock for modules
def get_mock_module():
    m = MagicMock()
    m.__path__ = []
    # Avoid recursion issues by not returning self in getattr if not needed,
    # but MagicMock handles it usually.
    return m


# Establish mocks for missing dependencies
sys.modules["anthropic"] = get_mock_module()
sys.modules["langsmith"] = get_mock_module()
sys.modules["langsmith.run_helpers"] = get_mock_module()
sys.modules["langsmith.runnables"] = get_mock_module()
sys.modules["langchain_tests"] = get_mock_module()
sys.modules["tenacity"] = get_mock_module()

from unittest.mock import patch

# Try importing real stuff
try:
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from langchain_anthropic import ChatAnthropic
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
except ImportError as e:
    print(f"FAILED to import libs: {e}")
    sys.exit(1)

try:
    from pydantic import BaseModel, Field
except ImportError:

    class BaseModel(MagicMock):
        pass

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
    try:
        model = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            api_key="sk-ant-fake-key",
            betas=["advanced-tool-use-2025-11-20"],
        )
    except Exception as e:
        print(f"Error initializing ChatAnthropic: {e}")
        return

    # Patch the _client property to return our mock
    with patch.object(ChatAnthropic, "_client", return_value=mock_client):
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

        try:
            # Call _get_request_payload directly
            payload = model._get_request_payload(
                input_=test_messages, cache_control={"type": "ephemeral", "ttl": "5m"}
            )

            # print("Payload generated successfully.")
            last_message = payload["messages"][-1]

            content_blocks = last_message["content"]
            last_block = content_blocks[-1]

            # Check the last block (the tool_use one)
            if "cache_control" in last_block:
                print("\n[FAIL] cache_control found in the last block!")
                if "caller" in last_block:
                    print(f"Block has caller: {last_block['caller']}")
                else:
                    print("Block has no caller.")
            else:
                print("\n[PASS] cache_control NOT found in the last block.")

        except Exception as e:
            print(f"Error during payload generation: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    reproduce()

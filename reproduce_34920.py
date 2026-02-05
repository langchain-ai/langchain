from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json

def test_reproduction():
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620", anthropic_api_key="fake")
    
    # Mock messages that should trigger the bug
    messages = [
        HumanMessage(content="What's the weather?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"location": "Boston"}, "id": "1"}]
        ),
        ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": "It's sunny.",
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            tool_call_id="1"
        )
    ]
    
    # In current master, this would likely produce a payload where cache_control 
    # is nested inside the tool_result content block.
    # On this PR branch, it should be moved to the tool_result level.
    
    payload = model._get_request_payload(messages)
    last_message = payload["messages"][-1]
    
    print("Formatted Message Payload:")
    print(json.dumps(last_message, indent=2))
    
    tool_result = last_message["content"][0]
    assert tool_result["type"] == "tool_result"
    
    # Check if cache_control is at the top level of tool_result
    if "cache_control" in tool_result:
        print("\n✅ SUCCESS: cache_control is at the top level of tool_result.")
    else:
        print("\n❌ FAILURE: cache_control is NOT at the top level of tool_result.")
        # Check if it's still inside content
        nested_content = tool_result["content"]
        if isinstance(nested_content, list) and "cache_control" in nested_content[0]:
            print("Found nested cache_control in content block.")

if __name__ == "__main__":
    try:
        test_reproduction()
    except Exception as e:
        print(f"Error: {e}")

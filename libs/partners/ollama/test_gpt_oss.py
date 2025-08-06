#!/usr/bin/env python
"""Test script to reproduce the gpt-oss:20b tool calling issue with ChatOllama."""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
    
    Returns:
        A string describing the weather.
    """
    return f"The weather in {location} is sunny and 72°F."


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate
    
    Returns:
        The result of the calculation as a string.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"


def test_gpt_oss_tool_calling():
    """Test tool calling with gpt-oss:20b model."""
    print("Testing ChatOllama with gpt-oss:20b model and tool calling...")
    print("-" * 60)
    
    # Initialize the model
    llm = ChatOllama(model="gpt-oss:20b", temperature=0)
    print(f"✓ Initialized ChatOllama with model: {llm.model}")
    
    # Define tools
    tools = [get_weather, calculate]
    print(f"✓ Defined {len(tools)} tools: {[t.name for t in tools]}")
    
    # Bind tools to the model
    try:
        llm_with_tools = llm.bind_tools(tools=tools)
        print("✓ Successfully bound tools to the model")
    except Exception as e:
        print(f"✗ Error binding tools: {e}")
        return False
    
    # Test queries that should trigger tool use
    test_queries = [
        "What's the weather like in San Francisco, CA?",
        "Calculate 42 * 17 for me",
        "What is 2 + 2?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)
        
        try:
            # Invoke the model with tools
            response = llm_with_tools.invoke(query)
            print(f"✓ Response received: {response}")
            
            # Check if tool calls were made
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"✓ Tool calls detected: {len(response.tool_calls)}")
                for tool_call in response.tool_calls:
                    print(f"  - Tool: {tool_call.get('name', 'unknown')}")
                    print(f"    Args: {tool_call.get('args', {})}")
            else:
                print("ℹ No tool calls in response")
                
        except Exception as e:
            print(f"✗ Error during invocation: {type(e).__name__}: {e}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    return True


def test_without_tools():
    """Test basic functionality without tools to ensure model works."""
    print("\nTesting basic ChatOllama without tools...")
    print("-" * 60)
    
    try:
        llm = ChatOllama(model="gpt-oss:20b", temperature=0)
        response = llm.invoke("Hello, how are you?")
        print(f"✓ Basic invocation works: {response}")
        return True
    except Exception as e:
        print(f"✗ Basic invocation failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ChatOllama gpt-oss:20b Tool Calling Test")
    print("=" * 60)
    
    # First test without tools to ensure basic functionality
    basic_works = test_without_tools()
    
    if basic_works:
        print("\n✓ Basic functionality confirmed. Testing tool calling...")
        # Now test with tools
        success = test_gpt_oss_tool_calling()
        
        if not success:
            print("\n⚠️  Tool calling test failed!")
            print("This confirms the issue described in the bug report.")
            print("\nExpected error:")
            print("  template: :108:130: executing \"\" at <index $prop.Type 0>:")
            print("  error calling index: reflect: slice index out of range")
    else:
        print("\n⚠️  Basic functionality test failed!")
        print("Please ensure:")
        print("  1. Ollama is running (ollama serve)")
        print("  2. gpt-oss:20b model is pulled (ollama pull gpt-oss:20b)")

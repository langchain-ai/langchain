#!/usr/bin/env python3

from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

def test_original_bug():
    """Test the original bug scenario"""
    print("Testing original bug scenario...")
    
    result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
        {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
        {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
    ]}))]

    parser = JsonOutputKeyToolsParser(key_name="func", first_tool_only=True, return_id=True)
    output = parser.parse_result(result)
    
    print(f"Output: {output}")
    print(f"Type: {type(output)}")
    
    if output is None:
        print("❌ BUG STILL EXISTS: Output is None")
        return False
    else:
        print("✅ BUG FIXED: Output is not None")
        print(f"Tool type: {output.get('type', 'N/A')}")
        print(f"Tool args: {output.get('args', 'N/A')}")
        return True

if __name__ == "__main__":
    test_original_bug()
 EOF
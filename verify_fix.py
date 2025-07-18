#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'core'))

from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

def test_fix():
    """Test the fix with explicit imports"""
    print("=== VERIFYING FIX ===")
    
    result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
        {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
        {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
    ]}))]

    print("Creating parser...")
    parser = JsonOutputKeyToolsParser(key_name="func", first_tool_only=True, return_id=True)
    print(f"Parser created: {parser}")
    print(f"Parser key_name: {parser.key_name}")
    print(f"Parser first_tool_only: {parser.first_tool_only}")
    print(f"Parser return_id: {parser.return_id}")
    
    print("
    output = parser.parse_result(result)
    
    print(f"Output: {output}")
    print(f"Type: {type(output)}")
    
    if output is None:
        print("❌ BUG STILL EXISTS: Output is None")
    else:
        print("✅ BUG FIXED: Output is not None")
        print(f"Tool type: {output.get('type', 'N/A')}")
        print(f"Tool args: {output.get('args', 'N/A')}")

if __name__ == "__main__":
    test_fix()
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

# Test the original bug scenario
result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
    {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
    {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
]}))]

parser = JsonOutputKeyToolsParser(key_name='func', first_tool_only=True, return_id=True)
output = parser.parse_result(result)

print('Output:', output)
print('Expected: Should return the func tool call, not None')

# Test with return_id=False
parser2 = JsonOutputKeyToolsParser(key_name='func', first_tool_only=True, return_id=False)
output2 = parser2.parse_result(result)
print('Output (return_id=False):', output2)
print('Expected: Should return {"a": 1}')


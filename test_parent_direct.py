#!/usr/bin/env python3

import sys
import importlib

# Force reload of the module
if 'langchain_core.output_parsers.openai_tools' in sys.modules:
    importlib.reload(sys.modules['langchain_core.output_parsers.openai_tools'])

from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

def test_with_fresh_import():
    """Test with fresh import to ensure our changes are loaded"""
    print("=== TESTING WITH FRESH IMPORT ===")
    
    result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
        {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
        {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
    ]}))]

    print("Creating parser...")
    parser = JsonOutputKeyToolsParser(key_name="func", first_tool_only=True, return_id=True)
    print(f"Parser: {parser}")
    print(f"Parser class: {parser.__class__}")
    print(f"Parser module: {parser.__class__.__module__}")
    
    # Check if our method is being used
    print(f"parse_result method: {parser.parse_result}")
    print(f"parse_result method module: {parser.parse_result.__func__.__module__}")
    
    print("
    try:
        output = parser.parse_result(result)
        print(f"Output: {output}")
        print(f"Type: {type(output)}")
        
        if output is None:
            print("❌ BUG STILL EXISTS: Output is None")
        else:
            print("✅ BUG FIXED: Output is not None")
            print(f"Tool type: {output.get('type', 'N/A')}")
            print(f"Tool args: {output.get('args', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_fresh_import()
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

# Test the original bug scenario
result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
    {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
    {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
]}))]

# Create a JsonOutputToolsParser instance and manually set first_tool_only=False
parser = JsonOutputToolsParser(first_tool_only=False, return_id=True)
all_results = parser.parse_result(result)
print('All results from parent:', all_results)

# Now filter manually
filtered = [res for res in all_results if res['type'] == 'func']
print('Filtered results:', filtered)

# Apply first_tool_only logic
if filtered:
    first_match = filtered[0]
    print('First match:', first_match)
else:
    print('No matches found')


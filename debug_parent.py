#!/usr/bin/env python3

from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser, JsonOutputToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

def test_parent_directly():
    """Test calling parent class directly to understand the data flow"""
    print("=== TESTING PARENT CLASS DIRECTLY ===")
    
    result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
        {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
        {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
    ]}))]

    print("1. Parent class with first_tool_only=False:")
    parent_parser = JsonOutputToolsParser(first_tool_only=False, return_id=True)
    parent_result = parent_parser.parse_result(result)
    print(f"   Result: {parent_result}")
    print(f"   Type: {type(parent_result)}")
    
    print("\n2. Parent class with first_tool_only=True:")
    parent_parser_first = JsonOutputToolsParser(first_tool_only=True, return_id=True)
    parent_result_first = parent_parser_first.parse_result(result)
    print(f"   Result: {parent_result_first}")
    print(f"   Type: {type(parent_result_first)}")
    
    print("\n3. Our child class:")
    child_parser = JsonOutputKeyToolsParser(key_name="func", first_tool_only=True, return_id=True)
    child_result = child_parser.parse_result(result)
    print(f"   Result: {child_result}")
    print(f"   Type: {type(child_result)}")

if __name__ == "__main__":
    test_parent_directly()
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

# Test what the parent parser returns
result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
    {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
    {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
]}))]

# Test parent parser with first_tool_only=False
parent_parser = JsonOutputToolsParser(first_tool_only=False, return_id=True)
parent_result = parent_parser.parse_result(result)
print('Parent parser (first_tool_only=False):', parent_result)

# Test parent parser with first_tool_only=True
parent_parser2 = JsonOutputToolsParser(first_tool_only=True, return_id=True)
parent_result2 = parent_parser2.parse_result(result)
print('Parent parser (first_tool_only=True):', parent_result2)


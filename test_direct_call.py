#!/usr/bin/env python3

from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

print('=== TESTING DIRECT CALL ===')

# Test the original bug scenario
result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
    {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
    {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
]}))]

parser = JsonOutputKeyToolsParser(key_name='func', first_tool_only=True, return_id=True)
print(f'Parser class: {parser.__class__}')
print(f'Parser method: {parser.parse_result}')

print('\nCalling parse_result...')
output = parser.parse_result(result)

print(f'\nFinal output: {output}')
print(f'Output type: {type(output)}')

if output is not None and output.get('type') == 'func' and output.get('args') == {'a': 1}:
    print('✅ BUG FIXED: Parser correctly returns the matching tool call')
else:
    print('❌ BUG STILL EXISTS: Parser returned None or incorrect result')

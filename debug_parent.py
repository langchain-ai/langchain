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

from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

# Test the original bug scenario
result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
    {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
    {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
]}))]

parser = JsonOutputKeyToolsParser(key_name='func', first_tool_only=True, return_id=True)

# Let's debug step by step
print('Input result:', result[0].message.additional_kwargs)

# First, let's see what the parent parser returns
temp_parser = JsonOutputKeyToolsParser(key_name='func', first_tool_only=False, return_id=True)
all_results = temp_parser.parse_result(result)
print('All parsed results:', all_results)

# Now test the fixed parser
output = parser.parse_result(result)
print('Final output:', output)

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

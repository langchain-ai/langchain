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

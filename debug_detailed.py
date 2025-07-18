from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

# Test the original bug scenario
result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
    {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
    {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
]}))]

class DebugJsonOutputKeyToolsParser(JsonOutputKeyToolsParser):
    def parse_result(self, result, *, partial=False):
        print(f'Starting parse_result with key_name={self.key_name}, first_tool_only={self.first_tool_only}')
        
        # Get all tool calls first
        temp_first_tool_only = self.first_tool_only
        self.first_tool_only = False
        try:
            parsed_result = super().parse_result(result, partial=partial)
            print(f'All parsed results: {parsed_result}')
        finally:
            self.first_tool_only = temp_first_tool_only
        
        # Filter by key_name first
        filtered_results = [res for res in parsed_result if res['type'] == self.key_name]
        print(f'Filtered results: {filtered_results}')

        if temp_first_tool_only:
            # Apply first_tool_only logic to the filtered results
            single_result = (
                filtered_results[0] if filtered_results else None
            )
            print(f'Single result: {single_result}')
            if self.return_id:
                return single_result
            if single_result:
                return single_result['args']
            return None
        
        # Return all filtered results
        if not self.return_id:
            filtered_results = [res['args'] for res in filtered_results]
        return filtered_results

parser = DebugJsonOutputKeyToolsParser(key_name='func', first_tool_only=True, return_id=True)
output = parser.parse_result(result)
print('Final output:', output)

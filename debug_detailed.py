#!/usr/bin/env python3

from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser, JsonOutputToolsParser
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

def debug_step_by_step():
    """Debug the fix step by step"""
    print("=== DEBUGGING STEP BY STEP ===")
    
    result = [ChatGeneration(message=AIMessage(content='', additional_kwargs={'tool_calls': [
        {'function': {'name': 'other', 'arguments': '{"b":2}'}, 'type': 'other'},
        {'function': {'name': 'func', 'arguments': '{"a":1}'}, 'type': 'func'}
    ]}))]

    print("1. Testing parent class behavior:")
    parent_parser = JsonOutputToolsParser(first_tool_only=False, return_id=True)
    parent_result = parent_parser.parse_result(result)
    print(f"   Parent result (all tools): {parent_result}")
    
    parent_parser_first = JsonOutputToolsParser(first_tool_only=True, return_id=True)
    parent_result_first = parent_parser_first.parse_result(result)
    print(f"   Parent result (first only): {parent_result_first}")
    
    print("\n2. Testing our fixed parser:")
    parser = JsonOutputKeyToolsParser(key_name="func", first_tool_only=True, return_id=True)
    
    # Let's manually trace through our logic
    print("   Step 2a: Getting all tool calls...")
    temp_first_tool_only = parser.first_tool_only
    parser.first_tool_only = False
    parsed_result = JsonOutputToolsParser.parse_result(parser, result)
    parser.first_tool_only = temp_first_tool_only
    print(f"   All parsed results: {parsed_result}")
    
    print("   Step 2b: Filtering by key_name...")
    filtered_results = [res for res in parsed_result if res["type"] == parser.key_name]
    print(f"   Filtered results: {filtered_results}")
    
    print("   Step 2c: Final result...")
    final_result = parser.parse_result(result)
    print(f"   Final result: {final_result}")

if __name__ == "__main__":
    debug_step_by_step()
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


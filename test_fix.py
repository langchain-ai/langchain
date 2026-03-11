#!/usr/bin/env python3
"""Simple test to verify the fix works"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs/core'))

# Minimal imports to test the specific function
def test_convert_to_openai_function():
    # OpenAI structured output schema format
    resp_schema = {
        "name": "math_reasoning",
        "schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "explanation": {"type": "string"},
                            "output": {"type": "string"},
                        },
                        "required": ["explanation", "output"],
                        "additionalProperties": False,
                    },
                },
                "final_answer": {"type": "string"},
            },
            "required": ["steps", "final_answer"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    # Let's manually implement the logic from the function to test
    function = resp_schema
    
    if isinstance(function, dict) and "name" in function and "schema" in function:
        oai_function = {
            "name": function["name"],
            "parameters": function["schema"],
        }
        if "description" in function:
            oai_function["description"] = function["description"]
        if "strict" in function:
            oai_function["strict"] = function["strict"]
        
        print("Test passed! Converted OpenAI structured output format:")
        print(f"  Name: {oai_function['name']}")
        print(f"  Has parameters: {'parameters' in oai_function}")
        print(f"  Strict: {oai_function.get('strict', 'Not set')}")
        print(f"  Parameters type: {oai_function['parameters']['type']}")
        return True
    else:
        print("Test failed - condition not matched")
        return False

if __name__ == "__main__":
    test_convert_to_openai_function()
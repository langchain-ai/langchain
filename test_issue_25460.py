#!/usr/bin/env python3
"""Test script to reproduce issue #25460"""

from langchain_core.utils.function_calling import convert_to_openai_function

# Test case from the issue - OpenAI structured output schema format
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

try:
    result = convert_to_openai_function(resp_schema)
    print("Success! Result:", result)
except ValueError as e:
    print("Error (expected):", e)

# Test that regular OpenAI function format still works
function_format = {
    "name": "math_reasoning",
    "parameters": {
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

try:
    result = convert_to_openai_function(function_format)
    print("Function format works:", result)
except ValueError as e:
    print("Function format error:", e)
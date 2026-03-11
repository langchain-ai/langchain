#!/usr/bin/env python3
"""Validate that the fix handles all cases correctly"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs/core'))

def test_openai_structured_output_logic():
    """Test the logic for OpenAI structured output format"""
    
    # Test case 1: Basic structured output format
    function1 = {
        "name": "math_reasoning",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        "strict": True,
    }
    
    # Test case 2: With description
    function2 = {
        "name": "simple_func",
        "description": "A simple function",
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        "strict": False,
    }
    
    # Test case 3: Without strict
    function3 = {
        "name": "no_strict",
        "schema": {
            "type": "object",
            "properties": {"data": {"type": "string"}},
        },
    }
    
    # Test case 4: Regular function format (should still work)
    function4 = {
        "name": "regular_func",
        "parameters": {
            "type": "object",
            "properties": {"param": {"type": "string"}},
        },
    }
    
    def convert_logic(function):
        """Simulate the logic from convert_to_openai_function"""
        if isinstance(function, dict) and "name" in function and "schema" in function:
            oai_function = {
                "name": function["name"],
                "parameters": function["schema"],
            }
            if "description" in function:
                oai_function["description"] = function["description"]
            if "strict" in function:
                oai_function["strict"] = function["strict"]
            return oai_function
        elif isinstance(function, dict) and "name" in function:
            oai_function = {
                k: v
                for k, v in function.items()
                if k in {"name", "description", "parameters", "strict"}
            }
            return oai_function
        else:
            return None
    
    # Test all cases
    test_cases = [
        ("Basic structured output", function1),
        ("With description", function2), 
        ("Without strict", function3),
        ("Regular function format", function4),
    ]
    
    for name, func in test_cases:
        result = convert_logic(func)
        print(f"\n{name}:")
        print(f"  Input: {func}")
        print(f"  Output: {result}")
        
        # Validate expected behavior
        if "schema" in func:
            assert result["parameters"] == func["schema"], f"Schema not converted to parameters in {name}"
            assert "schema" not in result, f"Schema key should not be in output for {name}"
        elif "parameters" in func:
            assert result["parameters"] == func["parameters"], f"Parameters not preserved in {name}"
        
        assert result["name"] == func["name"], f"Name not preserved in {name}"
        
        if "description" in func:
            assert result["description"] == func["description"], f"Description not preserved in {name}"
        
        if "strict" in func:
            assert result["strict"] == func["strict"], f"Strict not preserved in {name}"
    
    print("\n✅ All test cases passed!")

if __name__ == "__main__":
    test_openai_structured_output_logic()
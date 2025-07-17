#!/usr/bin/env python3
"""Test script to verify the JSON schema dereference fix works correctly."""

import sys
import os

# Add the libs/core directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'core'))

from langchain_core.utils.json_schema import dereference_refs

def test_original_issue():
    """Test the original issue from the bug report."""
    print("Testing original issue...")
    
    schema = {
        "type": "object",
        "properties": {
            "payload": {                        
                "anyOf": [
                    {  # variant 0
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "const": "ONE"}  
                        },
                    },
                    {  # variant 1
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "const": "TWO"},  
                            "startDate": {                               
                                "type": "string",
                                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                            },
                            "endDate": {                                 
                                "$ref": "#/properties/payload/anyOf/1/properties/startDate"
                            },
                        },
                    },
                ]
            }
        },
    }
    
    try:
        result = dereference_refs(schema)
        print("✓ SUCCESS: Original issue is fixed!")
        
        # Verify that endDate now has the same schema as startDate
        end_date_schema = result['properties']['payload']['anyOf'][1]['properties']['endDate']
        start_date_schema = result['properties']['payload']['anyOf'][1]['properties']['startDate']
        
        if end_date_schema == start_date_schema:
            print("✓ SUCCESS: endDate correctly references startDate schema")
            print(f"  endDate schema: {end_date_schema}")
        else:
            print("✗ FAILURE: endDate schema doesn't match startDate")
            print(f"  endDate: {end_date_schema}")
            print(f"  startDate: {start_date_schema}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases to ensure robustness."""
    print("
    
    # Test out-of-bounds index
    try:
        schema_invalid = {
            "type": "object",
            "properties": {
                "data": {
                    "anyOf": [{"type": "string"}]
                },
                "invalid": {"$ref": "#/properties/data/anyOf/5"}
            }
        }
        dereference_refs(schema_invalid)
        print("✗ FAILURE: Out-of-bounds index should raise KeyError")
        return False
    except KeyError as e:
        if "anyOf/5" in str(e):
            print("✓ SUCCESS: Out-of-bounds index correctly raises KeyError")
        else:
            print(f"✗ FAILURE: Wrong error message: {e}")
            return False
    
    # Test dictionary numeric key (regression test)
    try:
        schema_dict = {
            "type": "object",
            "properties": {
                "error_400": {"$ref": "#/$defs/400"},
            },
            "$defs": {
                400: {
                    "type": "object",
                    "properties": {"description": "Bad Request"},
                },
            },
        }
        result = dereference_refs(schema_dict)
        expected_desc = {"type": "object", "properties": {"description": "Bad Request"}}
        if result['properties']['error_400'] == expected_desc:
            print("✓ SUCCESS: Dictionary numeric keys still work")
        else:
            print("✗ FAILURE: Dictionary numeric keys broken")
            return False
    except Exception as e:
        print(f"✗ FAILURE: Dictionary numeric key test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Running JSON Schema dereference fix verification tests...
    
    success = True
    success &= test_original_issue()
    success &= test_edge_cases()
    
    if success:
        print("
        sys.exit(0)
    else:
        print("
        sys.exit(1)


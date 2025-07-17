#!/usr/bin/env python3
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'core'))

from langchain_core.utils.json_schema import dereference_refs

schema = {
    "type": "object",
    "properties": {
        "payload": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "const": "ONE"}
                    },
                },
                {
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

print("Original schema:")
print(json.dumps(schema, indent=2))

print("\nTesting the fix...")
try:
    result = dereference_refs(schema)
    print("\nDereferenced schema:")
    print(json.dumps(result, indent=2))
    
    end_date = result['properties']['payload']['anyOf'][1]['properties']['endDate']
    start_date = result['properties']['payload']['anyOf'][1]['properties']['startDate']
    
    print("\nendDate schema:")
    print(json.dumps(end_date, indent=2))
    print("\nstartDate schema:")
    print(json.dumps(start_date, indent=2))
    
    if end_date == start_date:
        print("\nSUCCESS: Schemas match!")
    else:
        print("\nFAILURE: Schemas don't match")
        
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()

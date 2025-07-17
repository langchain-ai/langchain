#!/usr/bin/env python3
import sys
import os
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

print("Testing the fix...")
try:
    result = dereference_refs(schema)
    print("SUCCESS: Fix works correctly!")
    end_date = result['properties']['payload']['anyOf'][1]['properties']['endDate']
    start_date = result['properties']['payload']['anyOf'][1]['properties']['startDate']
    if end_date == start_date:
        print("SUCCESS: endDate correctly references startDate schema")
    else:
        print("FAILURE: schemas don't match")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()

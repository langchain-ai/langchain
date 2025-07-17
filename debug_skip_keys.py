#!/usr/bin/env python3
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'core'))

from langchain_core.utils.json_schema import dereference_refs, _infer_skip_keys

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

print("Testing skip keys inference...")
try:
    skip_keys = _infer_skip_keys(schema, schema)
    print(f"Skip keys: {skip_keys}")
    
    print("\nTesting with no skip keys...")
    result = dereference_refs(schema, skip_keys=())
    print("Dereferenced with no skip keys:")
    print(json.dumps(result, indent=2))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

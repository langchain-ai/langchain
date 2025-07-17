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
                            "pattern": r"^\\d{4}-\\d{2}-\\d{2}$",
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

print("Testing issue reproduction...")
try:
    result = dereference_refs(schema)
    print("Success! No error occurred.")
except Exception as e:
    print(f"Error: {e}")
    print("Issue confirmed - the bug still exists.")

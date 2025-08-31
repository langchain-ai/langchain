#!/usr/bin/env python3
"""Reproduction/validation for the old infinite recursion in dereference_refs."""

from langchain_core.utils.json_schema import dereference_refs


def main() -> None:
    schema_with_infinite_recursion = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {"self_ref": {"$ref": "#/properties/nested"}},
            }
        },
    }

    result = dereference_refs(schema_with_infinite_recursion)
    print("Handled without recursion:", isinstance(result, dict))


if __name__ == "__main__":
    main()

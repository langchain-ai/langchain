from typing import Any


def _resolve_schema_references(schema: Any, definitions: dict[str, Any]) -> Any:
    """
    Resolve the $ref keys in a JSON schema object using the provided definitions.
    """
    if isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = _resolve_schema_references(item, definitions)
    elif isinstance(schema, dict):
        if "$ref" in schema:
            ref_key = schema.pop("$ref").split("/")[-1]
            ref = definitions.get(ref_key, {})
            schema.update(ref)
        else:
            for key, value in schema.items():
                schema[key] = _resolve_schema_references(value, definitions)
    return schema


def _convert_schema(schema: dict) -> dict:
    props = {k: {"title": k, **v} for k, v in schema["properties"].items()}
    return {
        "type": "object",
        "properties": props,
        "required": schema.get("required", []),
    }


def get_llm_kwargs(function: dict) -> dict:
    """Return the kwargs for the LLMChain constructor.

    Args:
        function: The function to use.

    Returns:
        The kwargs for the LLMChain constructor.
    """
    return {"functions": [function], "function_call": {"name": function["name"]}}

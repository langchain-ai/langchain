from typing import TypedDict

from langchain.tools import BaseTool, StructuredTool


class FunctionDescription(TypedDict):
    """Representation of a callable function to the OpenAI API."""

    name: str
    """The name of the function."""
    description: str
    """A description of the function."""
    parameters: dict
    """The parameters of the function."""


def add_enum_properties(prop: dict, definitions: dict) -> None:
    """Add enum key to properties for openai_function from pydantic format."""
    if "$ref" in prop:
        # when Field is not assigned
        ref_def_key = prop["$ref"].split("/")[-1]
        ref_def = definitions.get(ref_def_key, {})
        if "enum" in ref_def:
            prop["enum"] = ref_def["enum"]
            del prop["$ref"]
    elif "allOf" in prop:
        # when Field is assigned
        if len(prop["allOf"]) == 1 and "$ref" in prop["allOf"][0]:
            ref_def_key = prop["allOf"][0]["$ref"].split("/")[-1]
            ref_def = definitions.get(ref_def_key, {})
            if "enum" in ref_def:
                prop["enum"] = ref_def["enum"]
                del prop["allOf"]
    elif "anyOf" in prop:
        # union of enums
        # only resolve enums if all $ref is enum
        aggregated_enums_set = set()
        for any_of_prop in prop["anyOf"]:
            if "$ref" in any_of_prop:
                ref_def_key = any_of_prop["$ref"].split("/")[-1]
                ref_def = definitions.get(ref_def_key, {})
                if "enum" in ref_def:
                    aggregated_enums_set.update(ref_def["enum"])
            else:
                break
        else:
            prop["enum"] = sorted(aggregated_enums_set)
            del prop["anyOf"]


def format_tool_to_openai_function(tool: BaseTool) -> FunctionDescription:
    """Format tool into the OpenAI function API."""
    if isinstance(tool, StructuredTool):
        schema_ = tool.args_schema.schema()
        # Bug with required missing for structured tools.
        required = sorted(schema_["properties"])  # BUG WORKAROUND

        for prop in schema_["properties"].values():
            add_enum_properties(prop=prop, definitions=schema_.get("definitions", {}))

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": schema_["properties"],
                "required": required,
            },
        }
    else:
        if tool.args_schema:
            parameters = tool.args_schema.schema()
        else:
            parameters = {
                # This is a hack to get around the fact that some tools
                # do not expose an args_schema, and expect an argument
                # which is a string.
                # And Open AI does not support an array type for the
                # parameters.
                "properties": {
                    "__arg1": {"title": "__arg1", "type": "string"},
                },
                "required": ["__arg1"],
                "type": "object",
            }

        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        }

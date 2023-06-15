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


def format_tool_to_openai_function(tool: BaseTool) -> FunctionDescription:
    """Format tool into the open AI function API."""
    if isinstance(tool, StructuredTool):
        schema_ = tool.args_schema.schema()
        # Bug with required missing for structured tools.
        required = sorted(schema_["properties"])  # BUG WORKAROUND
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

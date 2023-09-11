from langchain.tools import BaseTool
from langchain.utils.openai_functions import (
    FunctionDescription,
    convert_pydantic_to_openai_function,
)


def format_tool_to_openai_function(tool: BaseTool) -> FunctionDescription:
    """Format tool into the OpenAI function API."""
    if tool.args_schema:
        return convert_pydantic_to_openai_function(
            tool.args_schema, name=tool.name, description=tool.description
        )
    else:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
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
            },
        }

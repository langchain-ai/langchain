from langchain_community.utils.openai_functions import (
    FunctionDescription,
    ToolDescription,
    convert_pydantic_to_openai_function,
)
from langchain_core.tools import BaseTool, Tool, create_schema_from_function

from langchain.chains.base import Chain


def format_tool_to_openai_function(tool: BaseTool) -> FunctionDescription:
    """Format tool into the OpenAI function API."""
    args_schema = tool.args_schema
    if not args_schema:
        func = tool.func or tool.coroutine if isinstance(tool, Tool) else tool._run
        # in case the tool is a method of a Chain
        if hasattr(func, "__self__") and isinstance(func.__self__, Chain):
            args_schema = func.__self__.get_input_schema()
        else:
            args_schema = create_schema_from_function(f"{tool.name}Schema", func)
    return convert_pydantic_to_openai_function(
        args_schema, name=tool.name, description=tool.description
    )


def format_tool_to_openai_tool(tool: BaseTool) -> ToolDescription:
    """Format tool into the OpenAI function API."""
    function = format_tool_to_openai_function(tool)
    return {"type": "function", "function": function}

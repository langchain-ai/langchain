# these stubs are just for backwards compatibility

from langchain_core.utils.function_calling import (
    FunctionDescription,
    GigaFunctionDescription,
    ToolDescription,
    convert_pydantic_to_openai_function,
    convert_pydantic_to_openai_tool,
)

__all__ = [
    "FunctionDescription",
    "GigaFunctionDescription",
    "ToolDescription",
    "convert_pydantic_to_openai_function",
    "convert_pydantic_to_openai_tool",
]

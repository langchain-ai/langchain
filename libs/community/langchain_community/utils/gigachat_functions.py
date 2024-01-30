# these stubs are just for backwards compatibility

from langchain_core.utils.function_calling import (
    FunctionDescription,
    ToolDescription,
    convert_pydantic_to_gigachat_function,
)

__all__ = [
    "FunctionDescription",
    "ToolDescription",
    "convert_pydantic_to_gigachat_function",
]

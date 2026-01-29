"""Tools are classes that an Agent uses to interact with the world.

Each tool has a description. Agent uses the description to choose the righ tool for the
job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.tools.base import (
        FILTERED_ARGS,
        ArgsSchema,
        BaseTool,
        BaseToolkit,
        InjectedToolArg,
        InjectedToolCallId,
        SchemaAnnotationError,
        ToolException,
        _get_runnable_config_param,
        create_schema_from_function,
    )
    from langchain_core.tools.convert import (
        convert_runnable_to_tool,
        tool,
    )
    from langchain_core.tools.render import (
        ToolsRenderer,
        render_text_description,
        render_text_description_and_args,
    )
    from langchain_core.tools.retriever import (
        RetrieverInput,
        create_retriever_tool,
    )
    from langchain_core.tools.simple import Tool
    from langchain_core.tools.structured import StructuredTool

__all__ = (
    "FILTERED_ARGS",
    "ArgsSchema",
    "BaseTool",
    "BaseToolkit",
    "InjectedToolArg",
    "InjectedToolCallId",
    "RetrieverInput",
    "SchemaAnnotationError",
    "StructuredTool",
    "Tool",
    "ToolException",
    "ToolsRenderer",
    "_get_runnable_config_param",
    "convert_runnable_to_tool",
    "create_retriever_tool",
    "create_schema_from_function",
    "render_text_description",
    "render_text_description_and_args",
    "tool",
)

_dynamic_imports = {
    "FILTERED_ARGS": "base",
    "ArgsSchema": "base",
    "BaseTool": "base",
    "BaseToolkit": "base",
    "InjectedToolArg": "base",
    "InjectedToolCallId": "base",
    "SchemaAnnotationError": "base",
    "ToolException": "base",
    "_get_runnable_config_param": "base",
    "create_schema_from_function": "base",
    "convert_runnable_to_tool": "convert",
    "tool": "convert",
    "ToolsRenderer": "render",
    "render_text_description": "render",
    "render_text_description_and_args": "render",
    "RetrieverInput": "retriever",
    "create_retriever_tool": "retriever",
    "Tool": "simple",
    "StructuredTool": "structured",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)

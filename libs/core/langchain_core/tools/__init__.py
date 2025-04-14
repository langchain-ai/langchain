"""**Tools** are classes that an Agent uses to interact with the world.

Each tool has a **description**. Agent uses the description to choose the right
tool for the job.

**Class hierarchy:**

.. code-block::

    RunnableSerializable --> BaseTool --> <name>Tool  # Examples: AIPluginTool, BaseGraphQLTool
                                          <name>      # Examples: BraveSearch, HumanInputRun

**Main helpers:**

.. code-block::

    CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
"""  # noqa: E501

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

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

__all__ = [
    "ArgsSchema",
    "BaseTool",
    "BaseToolkit",
    "FILTERED_ARGS",
    "SchemaAnnotationError",
    "ToolException",
    "InjectedToolArg",
    "InjectedToolCallId",
    "_get_runnable_config_param",
    "create_schema_from_function",
    "convert_runnable_to_tool",
    "tool",
    "ToolsRenderer",
    "render_text_description",
    "render_text_description_and_args",
    "RetrieverInput",
    "create_retriever_tool",
    "Tool",
    "StructuredTool",
]

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
    package = __spec__.parent
    if module_name == "__module__" or module_name is None:
        result = import_module(f".{attr_name}", package=package)
    else:
        module = import_module(f".{module_name}", package=package)
        result = getattr(module, attr_name)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)

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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from langchain_core.tools.base import (
        FILTERED_ARGS as FILTERED_ARGS,
    )
    from langchain_core.tools.base import (
        BaseTool as BaseTool,
    )
    from langchain_core.tools.base import (
        BaseToolkit as BaseToolkit,
    )
    from langchain_core.tools.base import (
        InjectedToolArg as InjectedToolArg,
    )
    from langchain_core.tools.base import InjectedToolCallId as InjectedToolCallId
    from langchain_core.tools.base import SchemaAnnotationError as SchemaAnnotationError
    from langchain_core.tools.base import (
        ToolException as ToolException,
    )
    from langchain_core.tools.base import (
        _get_runnable_config_param as _get_runnable_config_param,
    )
    from langchain_core.tools.base import (
        create_schema_from_function as create_schema_from_function,
    )
    from langchain_core.tools.convert import (
        convert_runnable_to_tool as convert_runnable_to_tool,
    )
    from langchain_core.tools.convert import tool as tool
    from langchain_core.tools.render import ToolsRenderer as ToolsRenderer
    from langchain_core.tools.render import (
        render_text_description as render_text_description,
    )
    from langchain_core.tools.render import (
        render_text_description_and_args as render_text_description_and_args,
    )
    from langchain_core.tools.retriever import RetrieverInput as RetrieverInput
    from langchain_core.tools.retriever import (
        create_retriever_tool as create_retriever_tool,
    )
    from langchain_core.tools.simple import Tool as Tool
    from langchain_core.tools.structured import StructuredTool as StructuredTool

def __getattr__(name: str) -> Any:
    if name == "FILTERED_ARGS":
        from langchain_core.tools.base import FILTERED_ARGS

        return FILTERED_ARGS
    if name == "BaseTool":
        from langchain_core.tools.base import BaseTool

        return BaseTool
    if name == "BaseToolkit":
        from langchain_core.tools.base import BaseToolkit

        return BaseToolkit
    if name == "InjectedToolArg":
        from langchain_core.tools.base import InjectedToolArg

        return InjectedToolArg
    if name == "InjectedToolCallId":
        from langchain_core.tools.base import InjectedToolCallId

        return InjectedToolCallId
    if name == "SchemaAnnotationError":
        from langchain_core.tools.base import SchemaAnnotationError

        return SchemaAnnotationError
    if name == "ToolException":
        from langchain_core.tools.base import ToolException

        return ToolException
    if name == "_get_runnable_config_param":
        from langchain_core.tools.base import _get_runnable_config_param

        return _get_runnable_config_param
    if name == "create_schema_from_function":
        from langchain_core.tools.base import create_schema_from_function

        return create_schema_from_function
    if name == "convert_runnable_to_tool":
        from langchain_core.tools.convert import convert_runnable_to_tool

        return convert_runnable_to_tool
    if name == "tool":
        from langchain_core.tools.convert import tool

        return tool
    if name == "ToolsRenderer":
        from langchain_core.tools.render import ToolsRenderer

        return ToolsRenderer
    if name == "render_text_description":
        from langchain_core.tools.render import render_text_description

        return render_text_description
    if name == "render_text_description_and_args":
        from langchain_core.tools.render import render_text_description_and_args

        return render_text_description_and_args
    if name == "RetrieverInput":
        from langchain_core.tools.retriever import RetrieverInput

        return RetrieverInput
    if name == "create_retriever_tool":
        from langchain_core.tools.retriever import create_retriever_tool

        return create_retriever_tool
    if name == "Tool":
        from langchain_core.tools.simple import Tool

        return Tool
    if name == "StructuredTool":
        from langchain_core.tools.structured import StructuredTool

        return StructuredTool
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "FILTERED_ARGS",
    "BaseTool",
    "BaseToolkit",
    "InjectedToolArg",
    "InjectedToolCallId",
    "SchemaAnnotationError",
    "ToolException",
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

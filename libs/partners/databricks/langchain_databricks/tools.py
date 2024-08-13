"""Databricks tools."""

from typing import Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool


class DatabricksInput(BaseModel):
    """Input schema for Databricks tool.

    This docstring is **not** part of what is sent to the model when performing tool
    calling. The Field default values and descriptions **are** part of what is sent to
    the model when performing tool calling.
    """

    # TODO: Add input args and descriptions.
    # a: int = Field(..., description="first number")
    # b: int = Field(0, description="second number")
    ...


class DatabricksTool(BaseTool):
    """Databricks tool.

    Setup:
        # TODO: Replace with relevant packages, env vars.
        Install ``langchain-databricks`` and set environment variable ``DATABRICKS_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-databricks
            export DATABRICKS_API_KEY="your-api-key"

    Instantiation:
        .. code-block:: python

            tool = DatabricksTool(
                # TODO: init params
            )

    Invocation with args:
        .. code-block:: python

            # TODO: invoke args
            tool.invoke({...})

        .. code-block:: python

            # TODO: output of invocation

    Invocation with ToolCall:

        .. code-block:: python

            # TODO: invoke args
            tool.invoke({"args": {...}, "id": "1", "name": tool.name, "type": "tool_call})

        .. code-block:: python

            # TODO: output of invocation
    """

    # TODO: Set tool name and description
    name: str = "TODO: Tool name"
    """The name that is passed to the model when performing tool calling."""
    description: str = "TODO: Tool description."
    """The description that is passed to the model when performing tool calling."""
    args_schema: Type[BaseModel] = DatabricksInput
    """The schema that is passed to the model when performing tool calling."""

    # TODO: Add any other init params for the tool.
    # param1: Optional[str]
    # """param1 determines foobar"""

    # TODO: Replaced *args with real tool arguments.
    def _run(
        self, *args, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        raise NotImplementedError

    # TODO: Implement if tool has native async functionality, otherwise delete.

    # async def _arun(
    #     self,
    #     *args,
    #     run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    # ) -> str:
    #     ...

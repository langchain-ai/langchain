from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field


class FallbackToolInput(BaseModel):
    """Schema for the input to the FallbackTool."""
    task_description: str = Field(
        description="A brief description of the task that the agent failed to complete."
    )


class FallbackTool(BaseTool):
    """Tool for gracefully terminating tasks when the agent cannot proceed."""

    name: str = "fallback"
    description: str = (
        "Use this tool when a task cannot be completed with the current resources or "
        "when an infinite loop is detected."
    )
    args_schema: Type[BaseModel] = FallbackToolInput
    return_direct: bool = True  # Stops the agent and returns the result directly to the user.
    handle_tool_error = True  # Enables error handling.

    def _run(
        self, task_description: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Sync implementation of the fallback tool with error handling."""
        try:
            # Business logic for gracefully terminating a task
            return (
                f"I'm unable to complete the task: '{task_description}'. "
                "Please provide additional resources or adjust the query."
            )
        except Exception as e:
            # Log and handle unexpected errors
            raise ToolException(f"FallbackTool encountered an error: {str(e)}") from e

    async def _arun(
        self,
        task_description: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async implementation of the fallback tool with error handling."""
        try:
            # Async-friendly logic; can reuse the sync implementation
            return self._run(task_description, run_manager=run_manager)
        except Exception as e:
            # Log and handle unexpected errors asynchronously
            raise ToolException(f"FallbackTool encountered an error: {str(e)}") from e
"""Interface for tools."""

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, tool
from typing_extensions import override


class InvalidTool(BaseTool):
    """Tool that is run when invalid tool name is encountered by agent."""

    name: str = "invalid_tool"
    """Name of the tool."""
    description: str = "Called when tool name is invalid. Suggests valid tool names."
    """Description of the tool."""

    @override
    def _run(
        self,
        requested_tool_name: str,
        available_tool_names: list[str],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Use the tool."""
        available_tool_names_str = ", ".join(list(available_tool_names))
        return (
            f"{requested_tool_name} is not a valid tool, "
            f"try one of [{available_tool_names_str}]."
        )

    @override
    async def _arun(
        self,
        requested_tool_name: str,
        available_tool_names: list[str],
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Use the tool asynchronously."""
        available_tool_names_str = ", ".join(list(available_tool_names))
        return (
            f"{requested_tool_name} is not a valid tool, "
            f"try one of [{available_tool_names_str}]."
        )


__all__ = ["InvalidTool", "tool"]

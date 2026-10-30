from typing import Any, Callable, Dict, List, Optional
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

class HumanInTheLoopMiddleware:
    """Middleware to enable human-in-the-loop workflows for tool execution
    by reviewing or modifying ToolMessages before proceeding.
    """

    def __init__(
        self,
        approval_callback: Optional[Callable[[ToolMessage], bool]] = None,
        modifier_callback: Optional[Callable[[ToolMessage], ToolMessage]] = None,
    ):
        self.approval_callback = approval_callback
        self.modifier_callback = modifier_callback

    def process_tool_message(self, tool_message: ToolMessage) -> ToolMessage:
        """Process and optionally intercept/modify the ToolMessage."""
        if self.approval_callback and not self.approval_callback(tool_message):
            # If human rejects, return a modified message or error state
            return ToolMessage(
                content="Tool execution rejected by human supervisor.",
                tool_call_id=tool_message.tool_call_id,
                status="error"
            )

        if self.modifier_callback:
            tool_message = self.modifier_callback(tool_message)

        return tool_message

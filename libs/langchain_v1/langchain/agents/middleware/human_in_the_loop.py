"""Human in the loop middleware."""

from typing import Any

from langchain_core.messages import ToolCall, ToolMessage
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
from langgraph.types import interrupt

from langchain.agents.middleware.types import AgentMiddleware, AgentState

ToolInterruptConfig = dict[str, HumanInterruptConfig]


class HumanInTheLoopMiddleware(AgentMiddleware):
    """Human in the loop middleware."""

    def __init__(
        self,
        tool_configs: ToolInterruptConfig,
        message_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            tool_configs: The tool interrupt configs to use for the middleware.
            message_prefix: The message prefix to use when constructing interrupt content.
        """
        super().__init__()
        self.tool_configs = tool_configs
        self.message_prefix = message_prefix

    def after_model(self, state: AgentState) -> dict[str, Any] | None:
        """Trigger HITL flows for relevant tool calls after an AIMessage."""
        messages = state["messages"]
        if not messages:
            return None

        last_message = messages[-1]

        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return None

        # Separate tool calls that need interrupts from those that don't
        interrupt_tool_calls = []
        auto_approved_tool_calls = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in self.tool_configs:
                interrupt_tool_calls.append(tool_call)
            else:
                auto_approved_tool_calls.append(tool_call)

        # If no interrupts needed, return early
        if not interrupt_tool_calls:
            return None

        # Process all tool calls that require interrupts
        approved_tool_calls: list[ToolCall] = auto_approved_tool_calls.copy()
        rejected_tool_calls: list[ToolCall] = []
        artificial_tool_messages: list[ToolMessage] = []

        # Create interrupt requests for all tools that need approval
        interrupt_requests: list[HumanInterrupt] = []
        for tool_call in interrupt_tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            description = f"{self.message_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"
            tool_config = self.tool_configs[tool_name]

            request: HumanInterrupt = {
                "action_request": ActionRequest(
                    action=tool_name,
                    args=tool_args,
                ),
                "config": tool_config,
                "description": description,
            }
            interrupt_requests.append(request)

        responses: list[HumanResponse] = interrupt(interrupt_requests)

        # TODO: map responses to tool call ids explicitly instead of assuming order
        # Right now this will fail if there's not a corresponding response for each tool call
        # Which we want to block on anyways but can do more gracefully
        for i, response in enumerate(responses):
            tool_call = interrupt_tool_calls[i]

            if response["type"] == "accept":
                approved_tool_calls.append(tool_call)
            elif response["type"] == "edit":
                edited: ActionRequest = response["args"]  # type: ignore[assignment]
                new_tool_call = ToolCall(
                    type="tool_call",
                    name=tool_call["name"],
                    args=edited["args"],
                    id=tool_call["id"],
                )
                approved_tool_calls.append(new_tool_call)
            elif response["type"] == "ignore":
                rejected_tool_calls.append(tool_call)
                ignore_message = ToolMessage(
                    content=f"User ignored the tool call for `{tool_name}` with id {tool_call['id']}",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    status="success",
                )
                artificial_tool_messages.append(ignore_message)
            elif response["type"] == "response":
                rejected_tool_calls.append(tool_call)
                tool_message = ToolMessage(
                    content=response["args"],  # type: ignore[assignment]
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    status="error",
                )
                artificial_tool_messages.append(tool_message)
            else:
                msg = f"Unknown response type: {response['type']}"
                raise ValueError(msg)

        last_message.tool_calls = [*approved_tool_calls, *rejected_tool_calls]  # type: ignore[assignment]

        if len(approved_tool_calls) > 0:
            return {"messages": [last_message, *artificial_tool_messages]}

        return {"jump_to": "model", "messages": artificial_tool_messages}

"""Human in the loop middleware."""

from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.types import interrupt
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState


class ActionRequest(TypedDict):
    """Request for human feedback.

    Attributes:
        message: Human-readable message describing what action is needed
        args: Args associated with the action (tool kwargs, for example)
        allowed_actions: List of allowed action types, can be one of "approve", "reject", "edit"
        description: Optional detailed description

        TODO: add args_schema to specify the allowed schema for the response args
    """

    message: str
    args: dict
    allowed_actions: list[Literal["approve", "reject", "edit"]]
    description: str | None


class ApproveResponse(TypedDict):
    """Response when a human approves the action."""

    action: Literal["approve"]
    """Type of action taken by the human, always 'approve'."""

    args: NotRequired[dict]
    """Edits to the args, if provided."""


class RejectResponse(TypedDict):
    """Response when a human rejects the action."""

    action: Literal["reject"]
    """Type of action taken by the human, always 'reject'."""

    message: NotRequired[str]
    """Reason for the rejection, if provided."""


ActionResponse = ApproveResponse | RejectResponse

AllowedActions = Literal["approve", "reject", "edit"]


class HumanInTheLoopMiddleware(AgentMiddleware):
    """Human in the loop middleware."""

    def __init__(
        self,
        tool_configs: dict[str, bool | list[AllowedActions]],
        *,
        action_request_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            tool_configs: Mapping of tool name to allowed actions.
                If a tool doesn't have an entry, it's auto-approved by default.
                * `True` indicates all actions are allowed: ["approve", "reject", "edit"].
                * `False` indicates that the tool is auto-approved.
                * List of actions indicates the specific actions allowed for this tool.
            action_request_prefix: The prefix to use when constructing action requests.
                This is used to provide context about the tool call and the action being requested.
        """
        super().__init__()
        resolved_tool_configs: dict[str, list[AllowedActions]] = {}
        for tool_name, tool_config in tool_configs.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_tool_configs[tool_name] = ["approve", "reject", "edit"]
            else:
                resolved_tool_configs[tool_name] = tool_config
        self.tool_configs = resolved_tool_configs
        self.action_request_prefix = action_request_prefix

    def after_model(self, state: AgentState) -> dict[str, Any] | None:
        """Trigger HITL flows for relevant tool calls after an AIMessage."""
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in messages if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Separate tool calls that need interrupts from those that don't
        interrupt_tool_calls: list[ToolCall] = []
        auto_approved_tool_calls = []

        for tool_call in last_ai_msg.tool_calls:
            interrupt_tool_calls.append(tool_call) if tool_call[
                "name"
            ] in self.tool_configs else auto_approved_tool_calls.append(tool_call)

        # If no interrupts needed, return early
        if not interrupt_tool_calls:
            return None

        # Process all tool calls that require interrupts
        approved_tool_calls: list[ToolCall] = auto_approved_tool_calls.copy()
        rejected_tool_calls: list[ToolCall] = []
        artificial_tool_messages: list[ToolMessage] = []

        # Create interrupt requests for all tools that need approval
        interrupt_requests: list[ActionRequest] = []
        for tool_call in interrupt_tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            message = f"{self.action_request_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"
            allowed_actions = self.tool_configs[tool_name]

            request: ActionRequest = {
                "message": message,
                "args": {"tool_name": tool_name, "tool_args": tool_args},
                "allowed_actions": allowed_actions,
                "description": message,
            }
            interrupt_requests.append(request)

        responses: list[ActionResponse] = interrupt(interrupt_requests)

        # Validate that the number of responses matches the number of interrupt tool calls
        if (responses_len := len(responses)) != (
            interrupt_tool_calls_len := len(interrupt_tool_calls)
        ):
            msg = (
                f"Number of human responses ({responses_len}) does not match "
                f"number of hanging tool calls ({interrupt_tool_calls_len})."
            )
            raise ValueError(msg)

        for i, response in enumerate(responses):
            tool_call = interrupt_tool_calls[i]
            allowed_actions = self.tool_configs[tool_call["name"]]

            if response["action"] == "approve" and "approve" in allowed_actions:
                # Use modified args if provided, otherwise use original args
                # need to raise if edits not supported, also support editing tool name
                if "args" in response:
                    if "edit" in allowed_actions:
                        edits = response["args"]
                        new_tool_call = ToolCall(
                            type="tool_call",
                            name=edits["tool_name"],
                            args=edits["tool_args"],
                            id=tool_call["id"],
                        )
                        approved_tool_calls.append(new_tool_call)
                    else:
                        error_msg = (
                            f"Unexpected human response: {response}. "
                            f"Response action 'approve' is not allowed "
                            f"for tool '{tool_call['name']}'. Expected one of "
                            f"{allowed_actions} based on the tool's configuration."
                        )
                        raise ValueError(error_msg)
                else:
                    approved_tool_calls.append(tool_call)
            elif response["action"] == "reject" and "reject" in allowed_actions:
                rejected_tool_calls.append(tool_call)
                # Use custom message if provided, otherwise use default
                if "message" in response:
                    content = response["message"]
                else:
                    content = (
                        f"User rejected the tool call for `{tool_call['name']}` "
                        f"with id {tool_call['id']}"
                    )

                tool_message = ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    status="error",
                )
                artificial_tool_messages.append(tool_message)
            else:
                msg = (
                    f"Unexpected human response: {response}. "
                    f"Response action '{response['action']}' "
                    f"is not allowed for tool '{tool_call['name']}'. "
                    f"Expected one of {allowed_actions} based on the tool's configuration."
                )
                raise ValueError(msg)

        last_ai_msg.tool_calls = [*approved_tool_calls, *rejected_tool_calls]

        if len(approved_tool_calls) > 0:
            return {"messages": [last_ai_msg, *artificial_tool_messages]}

        return {"jump_to": "model", "messages": artificial_tool_messages}

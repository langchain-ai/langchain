"""Human in the loop middleware."""

from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.types import interrupt
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState


class HumanInTheLoopConfig(TypedDict):
    """Configuration that defines what actions are allowed for a human interrupt.

    This controls the available interaction options when the graph is paused for human input.
    """

    allow_accept: NotRequired[bool]
    """Whether the human can approve the current action without changes."""
    allow_edit: NotRequired[bool]
    """Whether the human can approve the current action with edited content."""
    allow_respond: NotRequired[bool]
    """Whether the human can reject the current action with feedback."""


class ActionRequest(TypedDict):
    """Represents a request with a name and arguments."""

    action: str
    """The type or name of action being requested (e.g., "add_numbers")."""
    args: dict
    """Key-value pairs of arguments needed for the action (e.g., {"a": 1, "b": 2})."""


class HumanInTheLoopRequest(TypedDict):
    """Represents an interrupt triggered by the graph that requires human intervention.

    Example:
        ```python
        # Extract a tool call from the state and create an interrupt request
        request = HumanInterrupt(
            action_request=ActionRequest(
                action="run_command",  # The action being requested
                args={"command": "ls", "args": ["-l"]},  # Arguments for the action
            ),
            config=HumanInTheLoopConfig(
                allow_accept=True,  # Allow approval
                allow_respond=True,  # Allow rejection with feedback
                allow_edit=False,  # Don't allow approval with edits
            ),
            description="Please review the command before execution",
        )
        # Send the interrupt request and get the response
        response = interrupt([request])[0]
        ```
    """

    action_request: ActionRequest
    """The specific action being requested from the human."""
    config: HumanInTheLoopConfig
    """Configuration defining what response types are allowed."""
    description: str | None
    """Optional detailed description of what input is needed."""


class AcceptPayload(TypedDict):
    """Response when a human approves the action."""

    type: Literal["accept"]
    """The type of response when a human approves the action."""


class ResponsePayload(TypedDict):
    """Response when a human rejects the action."""

    type: Literal["response"]
    """The type of response when a human rejects the action."""

    args: NotRequired[str]
    """The message to be sent to the model explaining why the action was rejected."""


class EditPayload(TypedDict):
    """Response when a human edits the action."""

    type: Literal["edit"]
    """The type of response when a human edits the action."""

    args: ActionRequest
    """The action request with the edited content."""


HumanInTheLoopResponse = AcceptPayload | ResponsePayload | EditPayload
"""Aggregated response type for all possible human in the loop responses."""


class ToolConfig(TypedDict):
    """Configuration for a tool requiring human in the loop."""

    allow_accept: NotRequired[bool]
    """Whether the human can approve the current action without changes."""
    allow_edit: NotRequired[bool]
    """Whether the human can approve the current action with edited content."""
    allow_respond: NotRequired[bool]
    """Whether the human can reject the current action with feedback."""
    description: NotRequired[str]
    """The description attached to the request for human input."""


class HumanInTheLoopMiddleware(AgentMiddleware):
    """Human in the loop middleware."""

    def __init__(
        self,
        interrupt_on: dict[str, bool | ToolConfig],
        *,
        description_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            interrupt_on: Mapping of tool name to allowed actions.
                If a tool doesn't have an entry, it's auto-approved by default.
                * `True` indicates all actions are allowed: accept, edit, and respond.
                * `False` indicates that the tool is auto-approved.
                * ToolConfig indicates the specific actions allowed for this tool.
            description_prefix: The prefix to use when constructing action requests.
                This is used to provide context about the tool call and the action being requested.
                Not used if a tool has a description in its ToolConfig.
        """
        super().__init__()
        resolved_tool_configs: dict[str, ToolConfig] = {}
        for tool_name, tool_config in interrupt_on.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_tool_configs[tool_name] = ToolConfig(
                        allow_accept=True,
                        allow_edit=True,
                        allow_respond=True,
                    )
            elif any(
                tool_config.get(x, False) for x in ["allow_accept", "allow_edit", "allow_respond"]
            ):
                resolved_tool_configs[tool_name] = tool_config
        self.interrupt_on = resolved_tool_configs
        self.description_prefix = description_prefix

    def after_model(self, state: AgentState) -> dict[str, Any] | None:  # type: ignore[override]
        """Trigger interrupt flows for relevant tool calls after an AIMessage."""
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Separate tool calls that need interrupts from those that don't
        interrupt_tool_calls: list[ToolCall] = []
        auto_approved_tool_calls = []

        for tool_call in last_ai_msg.tool_calls:
            interrupt_tool_calls.append(tool_call) if tool_call[
                "name"
            ] in self.interrupt_on else auto_approved_tool_calls.append(tool_call)

        # If no interrupts needed, return early
        if not interrupt_tool_calls:
            return None

        # Process all tool calls that require interrupts
        revised_tool_calls: list[ToolCall] = auto_approved_tool_calls.copy()
        artificial_tool_messages: list[ToolMessage] = []
        are_pending_tool_calls = False

        # Create interrupt requests for all tools that need approval
        interrupt_requests: list[HumanInTheLoopRequest] = []
        for tool_call in interrupt_tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            config = self.interrupt_on[tool_name]
            description = (
                config.get("description")
                or f"{self.description_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"
            )

            request: HumanInTheLoopRequest = {
                "action_request": ActionRequest(
                    action=tool_name,
                    args=tool_args,
                ),
                "config": config,
                "description": description,
            }
            interrupt_requests.append(request)

        responses: list[HumanInTheLoopResponse] = interrupt(interrupt_requests)

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
            config = self.interrupt_on[tool_call["name"]]

            if response["type"] == "accept" and config.get("allow_accept"):
                revised_tool_calls.append(tool_call)
                are_pending_tool_calls = True
            elif response["type"] == "edit" and config.get("allow_edit"):
                edited_action = response["args"]
                revised_tool_calls.append(
                    ToolCall(
                        type="tool_call",
                        name=edited_action["action"],
                        args=edited_action["args"],
                        id=tool_call["id"],
                    )
                )
                are_pending_tool_calls = True
            elif response["type"] == "response" and config.get("allow_respond"):
                # Create a tool message with the human's text response
                content = response.get("args") or (
                    f"User rejected the tool call for `{tool_call['name']}` "
                    f"with id {tool_call['id']}"
                )
                tool_message = ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    status="error",
                )
                revised_tool_calls.append(tool_call)
                artificial_tool_messages.append(tool_message)
            else:
                allowed_actions = [
                    action
                    for action in ["accept", "edit", "response"]
                    if config.get(f"allow_{'respond' if action == 'response' else action}")
                ]
                msg = (
                    f"Unexpected human response: {response}. "
                    f"Response action '{response.get('type')}' "
                    f"is not allowed for tool '{tool_call['name']}'. "
                    f"Expected one of {allowed_actions} based on the tool's configuration."
                )
                raise ValueError(msg)

        # Update the AI message to only include approved tool calls
        last_ai_msg.tool_calls = revised_tool_calls

        if are_pending_tool_calls:
            return {"messages": [last_ai_msg, *artificial_tool_messages]}

        return {"jump_to": "model", "messages": artificial_tool_messages}

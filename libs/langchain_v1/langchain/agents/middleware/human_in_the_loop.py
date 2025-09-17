"""Human in the loop middleware."""

from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.types import interrupt
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState


class HumanInTheLoopConfig(TypedDict):
    """Configuration that defines what actions are allowed for a human interrupt.

    This controls the available interaction options when the graph is paused for human input.

    Attributes:
        allow_accept: Whether the human can approve the current action without changes
        allow_respond: Whether the human can reject the current action with feedback
        allow_edit: Whether the human can approve the current action with edited content
    """

    allow_accept: NotRequired[bool]
    allow_edit: NotRequired[bool]
    allow_respond: NotRequired[bool]


class ActionRequest(TypedDict):
    """Represents a request with a name and arguments.

    Attributes:
        action: The type or name of action being requested (e.g., "add_numbers")
        args: Key-value pairs of arguments needed for the action (e.g., {"a": 1, "b": 2})
    """

    action: str
    args: dict


class HumanInTheLoopRequest(TypedDict):
    """Represents an interrupt triggered by the graph that requires human intervention.

    Attributes:
        action_request: The specific action being requested from the human
        config: Configuration defining what response types are allowed
        description: Optional detailed description of what input is needed
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
    config: HumanInTheLoopConfig
    description: str | None


class AcceptPayload(TypedDict):
    """Response when a human approves the action."""

    type: Literal["accept"]


class ResponsePayload(TypedDict):
    """Response when a human rejects the action."""

    type: Literal["response"]

    args: NotRequired[str]


class EditPayload(TypedDict):
    """Response when a human edits the action."""

    type: Literal["edit"]

    args: ActionRequest


HumanInTheLoopResponse = AcceptPayload | ResponsePayload | EditPayload


class ToolConfig(TypedDict):
    """Configuration for a tool requiring human in the loop.

    Attributes:
        allow_accept: Whether the human can approve the current action without changes
        allow_edit: Whether the human can approve the current action with edited content
        allow_respond: Whether the human can reject the current action with feedback
        description: The description attached to the request for human input
    """

    allow_accept: NotRequired[bool]
    allow_edit: NotRequired[bool]
    allow_respond: NotRequired[bool]
    description: NotRequired[str]


class HumanInTheLoopMiddleware(AgentMiddleware):
    """Human in the loop middleware."""

    def __init__(
        self,
        tool_configs: dict[str, bool | ToolConfig],
        *,
        description_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            tool_configs: Mapping of tool name to allowed actions.
                If a tool doesn't have an entry, it's auto-approved by default.
                * `True` indicates all actions are allowed: accept, edit, and respond.
                * `False` indicates that the tool is auto-approved.
                * ToolConfig indicates the specific actions allowed for this tool.
            description_prefix: The prefix to use when constructing action requests.
                This is used to provide context about the tool call and the action being requested.
        """
        super().__init__()
        resolved_tool_configs: dict[str, ToolConfig] = {}
        for tool_name, tool_config in tool_configs.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_tool_configs[tool_name] = ToolConfig(
                        allow_accept=True,
                        allow_edit=True,
                        allow_respond=True,
                    )
            else:
                resolved_tool_configs[tool_name] = tool_config
        self.tool_configs = resolved_tool_configs
        self.description_prefix = description_prefix

    def after_model(self, state: AgentState) -> dict[str, Any] | None:
        """Trigger HITL flows for relevant tool calls after an AIMessage."""
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in messages if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Separate tool calls that need interrupts from those that don't
        hitl_tool_calls: list[ToolCall] = []
        auto_approved_tool_calls = []

        for tool_call in last_ai_msg.tool_calls:
            hitl_tool_calls.append(tool_call) if tool_call[
                "name"
            ] in self.tool_configs else auto_approved_tool_calls.append(tool_call)

        # If no interrupts needed, return early
        if not hitl_tool_calls:
            return None

        # Process all tool calls that require interrupts
        approved_tool_calls: list[ToolCall] = auto_approved_tool_calls.copy()
        artificial_tool_messages: list[ToolMessage] = []

        # Create interrupt requests for all tools that need approval
        hitl_requests: list[HumanInTheLoopRequest] = []
        for tool_call in hitl_tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            config = self.tool_configs[tool_name]
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
            hitl_requests.append(request)

        responses: list[HumanInTheLoopResponse] = interrupt(hitl_requests)

        # Validate that the number of responses matches the number of interrupt tool calls
        if (responses_len := len(responses)) != (hitl_tool_calls_len := len(hitl_tool_calls)):
            msg = (
                f"Number of human responses ({responses_len}) does not match "
                f"number of hanging tool calls ({hitl_tool_calls_len})."
            )
            raise ValueError(msg)

        for i, response in enumerate(responses):
            tool_call = hitl_tool_calls[i]
            config = self.tool_configs[tool_call["name"]]

            if response["type"] == "accept" and config.get("allow_accept"):
                approved_tool_calls.append(tool_call)
            elif response["type"] == "edit" and config.get("allow_edit"):
                edited_action = response["args"]
                approved_tool_calls.append(
                    ToolCall(
                        type="tool_call",
                        name=edited_action["action"],
                        args=edited_action["args"],
                        id=tool_call["id"],
                    )
                )
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
        last_ai_msg.tool_calls = approved_tool_calls

        if len(approved_tool_calls) > 0:
            return {"messages": [last_ai_msg, *artificial_tool_messages]}

        return {"jump_to": "model", "messages": artificial_tool_messages}

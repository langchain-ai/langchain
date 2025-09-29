"""Human in the loop middleware."""

from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.types import interrupt
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState


class Action(TypedDict):
    """Represents an action with a name and arguments."""

    name: str
    """The type or name of action being requested (e.g., "add_numbers")."""

    arguments: dict[str, Any]
    """Key-value pairs of arguments needed for the action (e.g., {"a": 1, "b": 2})."""

    description: NotRequired[str]
    """Description of the action to be reviewed, ex: the description for a tool."""


ResponseType = Literal["approve", "approve_with_edits", "reject"]


class ReviewConfig(TypedDict):
    """Policy for reviewing a HITL request."""

    allowed_responses: list[ResponseType]
    """The decisions that are allowed for this request."""

    description: NotRequired[str]
    """The description of the action to be reviewed."""

    arguments_schema: NotRequired[dict[str, Any]]
    """JSON schema for the arguments associated with the action."""


class HITLRequest(TypedDict):
    """Request for human feedback on a sequence of actions requested by a model."""

    action_requests: list[Action]
    """The specific actions being requested from the human."""

    review_configs: dict[str, ReviewConfig]
    """Configuration for the action review."""


class ApproveDecision(TypedDict):
    """Response when a human approves the action."""

    type: Literal["approve"]
    """The type of response when a human approves the action."""


class ApproveWithEditsDecision(TypedDict):
    """Response when a human approves the action with edits."""

    type: Literal["approve_with_edits"]
    """The type of response when a human approves the action with edits."""

    arguments: dict[str, Any]
    """The action request with the edited content."""

    message: NotRequired[str]
    """Optional rationale or notes."""


class RejectDecision(TypedDict):
    """Response when a human rejects the action."""

    type: Literal["reject"]
    """The type of response when a human rejects the action."""

    message: str
    """The message sent to the model explaining why the action was rejected."""


Decision = ApproveDecision | ApproveWithEditsDecision | RejectDecision


class HITLResponse(TypedDict):
    """Response payload for a HITLRequest."""

    decisions: list[Decision]
    """The decisions made by the human."""


class HumanInTheLoopMiddleware(AgentMiddleware):
    """Human in the loop middleware."""

    def __init__(
        self,
        interrupt_on: dict[str, bool | ReviewConfig],
        *,
        default_description: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            interrupt_on: Mapping of tool name to allowed actions.
                If a tool doesn't have an entry, it's auto-approved by default.
                * `True` indicates all actions are allowed: accept, edit, and respond.
                * `False` indicates that the tool is auto-approved.
                * ReviewConfig enables fine grained control over allowed actions:
                    * `allowed_responses`: The decisions that are allowed for this request.
                    * `description`: The description of the action to be reviewed.
                    * `arguments_schema`: JSON schema for the arguments associated with the action.
            default_description: The prefix to use when constructing action requests.
                This is used to provide context about the tool call and the action being requested.
                Not used if a tool has a description in its ToolConfig.
        """
        super().__init__()
        resolved_tool_configs: dict[str, ReviewConfig] = {}
        for tool_name, tool_config in interrupt_on.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_tool_configs[tool_name] = ReviewConfig(
                        allowed_responses=["approve", "approve_with_edits", "reject"],
                    )
            elif tool_config.get("allowed_responses", None):
                resolved_tool_configs[tool_name] = tool_config
        self.interrupt_on = resolved_tool_configs
        self.default_description = default_description

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
        approved_tool_calls: list[ToolCall] = auto_approved_tool_calls.copy()
        artificial_tool_messages: list[ToolMessage] = []

        # Create interrupt requests for all tools that need approval
        action_requests: list[Action] = []
        for tool_call in interrupt_tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            config = self.interrupt_on[tool_name]
            description = (
                config.get("description")
                or f"{self.default_description}\n\nTool: {tool_name}\nArgs: {tool_args}"
            )

            action_requests.append(
                {
                    "name": tool_name,
                    "arguments": tool_args,
                    "description": description,
                }
            )

        hitl_request: HITLRequest = {
            "action_requests": action_requests,
            "review_configs": self.interrupt_on,
        }

        response: HITLResponse = interrupt(hitl_request)

        # Validate that the number of responses matches the number of interrupt tool calls
        if (responses_len := len(response["decisions"])) != (
            interrupt_tool_calls_len := len(interrupt_tool_calls)
        ):
            msg = (
                f"Number of human responses ({responses_len}) does not match "
                f"number of hanging tool calls ({interrupt_tool_calls_len})."
            )
            raise ValueError(msg)

        for i, decision in enumerate(response["decisions"]):
            tool_call = interrupt_tool_calls[i]
            config = self.interrupt_on[tool_call["name"]]

            if decision["type"] == "approve" and "approve" in config["allowed_responses"]:
                approved_tool_calls.append(tool_call)
            elif (
                decision["type"] == "approve_with_edits"
                and "approve_with_edits" in config["allowed_responses"]
            ):
                edited_action = decision["arguments"]
                approved_tool_calls.append(
                    ToolCall(
                        type="tool_call",
                        name=edited_action["name"],
                        args=edited_action["args"],
                        id=tool_call["id"],
                    )
                )
            elif decision["type"] == "reject" and "reject" in config["allowed_responses"]:
                # Create a tool message with the human's text response
                content = decision.get("message") or (
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
                    f"Response action '{response.get('type')}' "
                    f"is not allowed for tool '{tool_call['name']}'. "
                    f"Expected one of {config.get('allowed_responses')} "
                    "based on the tool's configuration."
                )
                raise ValueError(msg)

        # Update the AI message to only include approved tool calls
        last_ai_msg.tool_calls = approved_tool_calls

        if len(approved_tool_calls) > 0:
            return {"messages": [last_ai_msg, *artificial_tool_messages]}

        return {"jump_to": "model", "messages": artificial_tool_messages}

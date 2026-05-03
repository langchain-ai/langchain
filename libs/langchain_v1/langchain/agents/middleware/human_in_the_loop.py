"""Human in the loop middleware."""

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeAlias

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    StateT,
)


class Action(TypedDict):
    """Represents an action with a name and args."""

    name: str
    """The type or name of action being requested (e.g., `'add_numbers'`)."""

    args: dict[str, Any]
    """Key-value pairs of args needed for the action (e.g., `{"a": 1, "b": 2}`)."""


class ActionRequest(TypedDict):
    """Represents an action request with a name, args, and description."""

    name: str
    """The name of the action being requested."""

    args: dict[str, Any]
    """Key-value pairs of args needed for the action (e.g., `{"a": 1, "b": 2}`)."""

    description: NotRequired[str]
    """The description of the action to be reviewed."""


DecisionType = Literal["approve", "edit", "reject", "respond"]


class ReviewConfig(TypedDict):
    """Policy for reviewing a HITL request."""

    action_name: str
    """Name of the action associated with this review configuration."""

    allowed_decisions: list[DecisionType]
    """The decisions that are allowed for this request."""

    args_schema: NotRequired[dict[str, Any]]
    """JSON schema for the args associated with the action, if edits are allowed."""


class HITLRequest(TypedDict):
    """Request for human feedback on a sequence of actions requested by a model."""

    action_requests: list[ActionRequest]
    """A list of agent actions for human review."""

    review_configs: list[ReviewConfig]
    """Review configuration for all possible actions."""


class ApproveDecision(TypedDict):
    """Response when a human approves the action."""

    type: Literal["approve"]
    """The type of response when a human approves the action."""


class EditDecision(TypedDict):
    """Response when a human edits the action."""

    type: Literal["edit"]
    """The type of response when a human edits the action."""

    edited_action: Action
    """Edited action for the agent to perform.

    Ex: for a tool call, a human reviewer can edit the tool name and args.
    """


class RejectDecision(TypedDict):
    """Response when a human rejects the action."""

    type: Literal["reject"]
    """The type of response when a human rejects the action."""

    message: NotRequired[str]
    """The message sent to the model explaining why the action was rejected."""


class RespondDecision(TypedDict):
    """Response when a human answers on behalf of the tool, skipping execution.

    Used for "ask user" style tools whose real implementation is the human's
    response. The tool is not executed; instead, a synthetic `ToolMessage` with
    `status="success"` and the provided `message` is returned to the model.
    """

    type: Literal["respond"]
    """The type of response when a human responds on behalf of the tool."""

    message: str
    """Content of the synthetic `ToolMessage` returned to the model."""


Decision = ApproveDecision | EditDecision | RejectDecision | RespondDecision


RejectionResponseFactory: TypeAlias = Callable[[ToolCall, RejectDecision], ToolMessage]
"""Factory that builds the synthetic `ToolMessage` returned for a `reject` decision.

!!! warning "Alpha"

    This signature is alpha and subject to change or removal without notice.

Receives the original `ToolCall` and the human's `RejectDecision`; returns the
`ToolMessage` paired with the call. See `InterruptOnConfig.rejection_response`
for usage guidance and contract requirements.
"""

RejectionResponse: TypeAlias = ToolMessage | RejectionResponseFactory
"""Override for the `ToolMessage` produced by a `reject` decision.

!!! warning "Alpha"

    This signature is alpha and subject to change or removal without notice.

Either a pre-built `ToolMessage` template or a `RejectionResponseFactory`
callable. See `InterruptOnConfig.rejection_response` for full usage guidance.
"""


class HITLResponse(TypedDict):
    """Response payload for a HITLRequest."""

    decisions: list[Decision]
    """The decisions made by the human."""


class _DescriptionFactory(Protocol):
    """Callable that generates a description for a tool call."""

    def __call__(
        self, tool_call: ToolCall, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> str:
        """Generate a description for a tool call."""
        ...


class InterruptOnConfig(TypedDict):
    """Configuration for an action requiring human in the loop.

    This is the configuration format used in the `HumanInTheLoopMiddleware.__init__`
    method.
    """

    allowed_decisions: list[DecisionType]
    """The decisions that are allowed for this action."""

    description: NotRequired[str | _DescriptionFactory]
    """The description attached to the request for human input.

    Can be either:

    - A static string describing the approval request
    - A callable that dynamically generates the description based on agent state,
        runtime, and tool call information

    Example:
        ```python
        # Static string description
        config = ToolConfig(
            allowed_decisions=["approve", "reject"],
            description="Please review this tool execution"
        )

        # Dynamic callable description
        def format_tool_description(
            tool_call: ToolCall,
            state: AgentState,
            runtime: Runtime[ContextT]
        ) -> str:
            import json
            return (
                f"Tool: {tool_call['name']}\\n"
                f"Arguments:\\n{json.dumps(tool_call['args'], indent=2)}"
            )

        config = InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            description=format_tool_description
        )
        ```
    """
    args_schema: NotRequired[dict[str, Any]]
    """JSON schema for the args associated with the action, if edits are allowed."""

    rejection_response: NotRequired[RejectionResponse]
    """Override the synthetic `ToolMessage` produced when a human rejects this action.

    !!! warning "Alpha"

        This field is alpha and subject to change or removal without notice.

    Two forms, with deliberately asymmetric strictness:

    - A pre-built `ToolMessage` template — the middleware returns a copy with
        `tool_call_id` and `name` overwritten from the rejected call, so any
        values you set for those two fields on the template are treated as
        placeholders. Use this form when you want a generic message reused
        across many tools without threading per-call ids through.
    - A `RejectionResponseFactory` (a `(ToolCall, RejectDecision) -> ToolMessage`
        callable). The factory has access to the call, so it is responsible
        for setting `tool_call_id` and `name` correctly; mismatches raise
        `ValueError` and a non-`ToolMessage` return raises `TypeError`.

    Only meaningful when `'reject'` is in `allowed_decisions`; setting it
    otherwise raises at middleware construction.

    By default the middleware emits a `status="error"` `ToolMessage`. Override
    when you need custom copy, additional metadata for downstream consumers,
    or a different `status` — for example, models that interpret `error` as a
    transient failure and immediately re-emit the same tool call can be
    calmed down with `status="success"` and retry-discouraging content.

    Example:
        ```python
        # Suppress retries on providers that re-emit on `status="error"`.
        config = InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            rejection_response=lambda tool_call, decision: ToolMessage(
                content=(
                    f"`{tool_call['name']}` declined: "
                    f"{decision.get('message', 'no reason provided')}. "
                    "Do not retry."
                ),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="success",
                additional_kwargs={"hitl_decision": "reject"},
            ),
        )
        ```
    """


def _default_rejection_message(tool_call: ToolCall, decision: RejectDecision) -> ToolMessage:
    """Build the default `ToolMessage` returned when a tool call is rejected."""
    # An explicitly empty `decision["message"]` falls back to the canned text;
    # treating it as "no message" matches caller intent.
    content = decision.get("message") or (
        f"User rejected the tool call for `{tool_call['name']}` with id {tool_call['id']}"
    )
    return ToolMessage(
        content=content,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
        status="error",
    )


def _resolve_rejection_response(
    override: RejectionResponse | None,
    tool_call: ToolCall,
    decision: RejectDecision,
) -> ToolMessage:
    """Resolve a `rejection_response` override into a concrete `ToolMessage`.

    Three branches:

    - `None` — fall back to the default rejection message.
    - A `ToolMessage` template — return a copy with `tool_call_id` and `name`
        overwritten from the rejected call so providers can pair the result to
        the originating call without callers threading those fields through.
        The caller's template is not mutated.
    - A factory callable — invoke it and validate the result. The return must
        be a `ToolMessage` whose `tool_call_id` and `name` match the rejected
        call, so the `AIMessage` / `ToolMessage` pair stays well-formed for
        providers (Anthropic in particular rejects mismatched `tool_use_id` /
        `tool_result.tool_use_id`).

    Validation is intentionally asymmetric: the template branch *stamps* the
    identity fields (template is meant to be generic and reusable), while the
    factory branch *validates* them (the factory has the call in hand and is
    expected to set them correctly).
    """
    if override is None:
        return _default_rejection_message(tool_call, decision)
    if isinstance(override, ToolMessage):
        return override.model_copy(
            update={"tool_call_id": tool_call["id"], "name": tool_call["name"]}
        )
    message = override(tool_call, decision)
    # User-supplied callable: verify the runtime contract even though the
    # static type narrows to `ToolMessage`.
    if not isinstance(message, ToolMessage):
        msg = (  # type: ignore[unreachable]
            f"`rejection_response` callable for tool '{tool_call['name']}' returned "
            f"{type(message).__name__!r}; expected a `ToolMessage`."
        )
        raise TypeError(msg)
    if message.tool_call_id != tool_call["id"]:
        msg = (
            f"`rejection_response` callable for tool '{tool_call['name']}' returned a "
            f"`ToolMessage` with `tool_call_id={message.tool_call_id!r}`, but the "
            f"rejected call's id is {tool_call['id']!r}. The ids must match so "
            "providers can pair the tool result with the originating tool call."
        )
        raise ValueError(msg)
    if message.name != tool_call["name"]:
        msg = (
            f"`rejection_response` callable for tool '{tool_call['name']}' returned a "
            f"`ToolMessage` with `name={message.name!r}`, but the rejected call's "
            f"name is {tool_call['name']!r}. Names must match so the tool result "
            "is correctly attributed to the originating tool call."
        )
        raise ValueError(msg)
    return message


class HumanInTheLoopMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
    """Human in the loop middleware."""

    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        description_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            interrupt_on: Mapping of tool name to allowed actions.

                If a tool doesn't have an entry, it's auto-approved by default.

                * `True` indicates all decisions are allowed: approve, edit, reject,
                    and respond.
                * `False` indicates that the tool is auto-approved.
                * `InterruptOnConfig` indicates the specific decisions allowed for this
                    tool.

                    The `InterruptOnConfig` can include a `description` field (`str` or
                    `Callable`) for custom formatting of the interrupt description.
            description_prefix: The prefix to use when constructing action requests.

                This is used to provide context about the tool call and the action being
                requested.

                Not used if a tool has a `description` in its `InterruptOnConfig`.

        Raises:
            ValueError: If an `InterruptOnConfig` sets `rejection_response` without
                including `'reject'` in `allowed_decisions` (the override would
                never be used).
        """
        super().__init__()
        resolved_configs: dict[str, InterruptOnConfig] = {}
        for tool_name, tool_config in interrupt_on.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_configs[tool_name] = InterruptOnConfig(
                        allowed_decisions=["approve", "edit", "reject", "respond"]
                    )
            elif tool_config.get("allowed_decisions"):
                if (
                    tool_config.get("rejection_response") is not None
                    and "reject" not in tool_config["allowed_decisions"]
                ):
                    msg = (
                        f"Tool '{tool_name}' has a `rejection_response` configured but "
                        "'reject' is not in `allowed_decisions`; the override would "
                        "never be used. Add 'reject' to `allowed_decisions` or remove "
                        "`rejection_response`."
                    )
                    raise ValueError(msg)
                resolved_configs[tool_name] = tool_config
        self.interrupt_on = resolved_configs
        self.description_prefix = description_prefix

    def _create_action_and_config(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> tuple[ActionRequest, ReviewConfig]:
        """Create an ActionRequest and ReviewConfig for a tool call."""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Generate description using the description field (str or callable)
        description_value = config.get("description")
        if callable(description_value):
            description = description_value(tool_call, state, runtime)
        elif description_value is not None:
            description = description_value
        else:
            description = f"{self.description_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"

        # Create ActionRequest with description
        action_request = ActionRequest(
            name=tool_name,
            args=tool_args,
            description=description,
        )

        # Create ReviewConfig
        # eventually can get tool information and populate args_schema from there
        review_config = ReviewConfig(
            action_name=tool_name,
            allowed_decisions=config["allowed_decisions"],
        )

        return action_request, review_config

    @staticmethod
    def _process_decision(
        decision: Decision,
        tool_call: ToolCall,
        config: InterruptOnConfig,
    ) -> tuple[ToolCall | None, ToolMessage | None]:
        """Process a single decision and return the revised tool call and optional tool message."""
        allowed_decisions = config["allowed_decisions"]

        if decision["type"] == "approve" and "approve" in allowed_decisions:
            return tool_call, None
        if decision["type"] == "edit" and "edit" in allowed_decisions:
            edited_action = decision["edited_action"]
            return (
                ToolCall(
                    type="tool_call",
                    name=edited_action["name"],
                    args=edited_action["args"],
                    id=tool_call["id"],
                ),
                None,
            )
        if decision["type"] == "reject" and "reject" in allowed_decisions:
            tool_message = _resolve_rejection_response(
                config.get("rejection_response"), tool_call, decision
            )
            return tool_call, tool_message
        if decision["type"] == "respond" and "respond" in allowed_decisions:
            # Skip tool execution; the human answers on behalf of the tool.
            tool_message = ToolMessage(
                content=decision["message"],
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="success",
            )
            return tool_call, tool_message
        msg = (
            f"Unexpected human decision: {decision}. "
            f"Decision type '{decision.get('type')}' "
            f"is not allowed for tool '{tool_call['name']}'. "
            f"Expected one of {allowed_decisions} based on the tool's configuration."
        )
        raise ValueError(msg)

    def after_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Trigger interrupt flows for relevant tool calls after an `AIMessage`.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Updated message with the revised tool calls.

        Raises:
            ValueError: If the number of human decisions does not match the number of
                interrupted tool calls.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Create action requests and review configs for tools that need approval
        action_requests: list[ActionRequest] = []
        review_configs: list[ReviewConfig] = []
        interrupt_indices: list[int] = []

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if (config := self.interrupt_on.get(tool_call["name"])) is not None:
                action_request, review_config = self._create_action_and_config(
                    tool_call, config, state, runtime
                )
                action_requests.append(action_request)
                review_configs.append(review_config)
                interrupt_indices.append(idx)

        # If no interrupts needed, return early
        if not action_requests:
            return None

        # Create single HITLRequest with all actions and configs
        hitl_request = HITLRequest(
            action_requests=action_requests,
            review_configs=review_configs,
        )

        # Send interrupt and get response
        decisions = interrupt(hitl_request)["decisions"]

        # Validate that the number of decisions matches the number of interrupt tool calls
        if (decisions_len := len(decisions)) != (interrupt_count := len(interrupt_indices)):
            msg = (
                f"Number of human decisions ({decisions_len}) does not match "
                f"number of hanging tool calls ({interrupt_count})."
            )
            raise ValueError(msg)

        # Process decisions and rebuild tool calls in original order
        revised_tool_calls: list[ToolCall] = []
        artificial_tool_messages: list[ToolMessage] = []
        decision_idx = 0

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if idx in interrupt_indices:
                # This was an interrupt tool call - process the decision
                config = self.interrupt_on[tool_call["name"]]
                decision = decisions[decision_idx]
                decision_idx += 1

                revised_tool_call, tool_message = self._process_decision(
                    decision, tool_call, config
                )
                if revised_tool_call is not None:
                    revised_tool_calls.append(revised_tool_call)
                if tool_message:
                    artificial_tool_messages.append(tool_message)
            else:
                # This was auto-approved - keep original
                revised_tool_calls.append(tool_call)

        # Update the AI message to only include approved tool calls
        last_ai_msg.tool_calls = revised_tool_calls

        return {"messages": [last_ai_msg, *artificial_tool_messages]}

    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async trigger interrupt flows for relevant tool calls after an `AIMessage`.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Updated message with the revised tool calls.
        """
        return self.after_model(state, runtime)

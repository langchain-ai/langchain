"""Human in the loop middleware."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Protocol,
    Union,
    cast,
    get_args,
)

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.runtime import get_runtime
from langgraph.types import Command, interrupt
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    ValidatorFunctionWrapHandler,
    WithJsonSchema,
    WrapValidator,
    create_model,
)
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
    ResponseT,
    StateT,
    ToolCallRequest,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime


_EDITED_TOOL_CALLS_KEY = "hitl_edited_tool_calls"
"""State key mapping tool call ID to the reviewer's replacement for it."""

_EDIT_NOTICE = (
    "Note: a human reviewer replaced this tool call before it ran. The call recorded in "
    "your message is the one you produced, not the one that executed. This was "
    "intentional and authorized. Do not re-issue your original call."
)
"""Default text prepended to the result of a tool call a reviewer edited."""


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
    """The human-provided reason for rejecting the action.

    The reason is framed as a user rejection when sent to the model. If omitted,
    the model is told that the tool was not executed and should not retry the same
    tool call unless the user asks for it.
    """


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


InterruptMode = Literal["batched", "per_call"]
"""How HITL pauses: one interrupt per model turn (`batched`) or per gated tool call."""


class ToolApprovalRequest(TypedDict):
    """Interrupt value raised once per gated tool call in `per_call` mode.

    Takes the place of `HITLRequest`. Answer it with a single `Decision`, keyed by the
    interrupt's ID, not a `HITLResponse`. The interrupt's `response_schema` lists the
    decisions allowed for the tool and, for edits, the tool's argument schema.
    """

    type: Literal["tool_approval"]
    """Always `"tool_approval"`; tells clients how to read this interrupt."""

    tool_call_id: str
    """ID of the model's tool call this approval is about."""

    name: str
    """Tool name, as the model requested it."""

    args: dict[str, Any]
    """Tool arguments, as the model requested them."""

    description: str
    """Text shown to the reviewer."""


def _edit_args(tool: BaseTool | None) -> object:
    """What an edit's `args` must look like: the tool's Pydantic schema, if it has one.

    That's the schema the model sees (`tool_call_schema`): it checks arg types and
    required args, and rejects args it doesn't declare unless the tool accepts extras.
    Validator methods on the tool's `args_schema` aren't part of it; they run when the
    tool does.
    A JSON-schema tool's schema is shown but not enforced, and with no tool any object
    is accepted.
    """
    schema = tool.tool_call_schema if tool else None
    if not (tool and isinstance(schema, type) and issubclass(schema, BaseModel)):
        shown = schema if isinstance(schema, dict) else {"type": "object"}
        return Annotated[dict[str, Any], WithJsonSchema(shown)]
    # `tool_call_schema` drops the tool's own `extra` setting, so read it from `args_schema`.
    allows_extra = getattr(tool.args_schema, "model_config", {}).get("extra") == "allow"
    # A subclass keeps the tool's fields, name and description.
    return create_model(
        schema.__name__,
        __base__=schema,
        __doc__=schema.__doc__,
        __cls_kwargs__={"extra": "allow" if allows_extra else "forbid"},
    )


def _as_sent(answer: object, check: ValidatorFunctionWrapHandler) -> object:
    """Check `answer`, then pass it on as sent, so the tool validates its args once."""
    check(answer)
    return answer


def _edit_decision(name: str, tool: BaseTool | None) -> object:
    """`EditDecision` for one tool: `name` pinned to it, `args` from `_edit_args`.

    Undeclared fields are rejected, so a typo fails instead of being dropped.
    """
    edited_action = create_model(
        "EditedAction",
        __config__=ConfigDict(extra="forbid"),
        name=(Literal[name], ...),
        args=(_edit_args(tool), ...),
    )
    decision = create_model(
        "EditDecision",
        __config__=ConfigDict(extra="forbid"),
        type=(Literal["edit"], ...),
        edited_action=(edited_action, ...),
    )
    return Annotated[decision, WrapValidator(_as_sent)]


def _decision_schema(
    allowed: Sequence[DecisionType], name: str, tool: BaseTool | None = None
) -> type[Decision]:
    """The per-call `response_schema`: today's decision types, limited to `allowed`.

    The edit branch is built for this tool: its name is pinned, so an edit can't switch
    tools, and its args follow the tool's own schema. An answer is checked only against
    the branch its `type` names, so a bad one gets a single error about what's wrong.
    A one-decision tool gets that decision's plain object schema.

    Returns a type rather than a JSON schema because LangGraph checks answers only
    against Python types; it publishes the type's JSON schema to clients.
    """
    by_type: dict[DecisionType, object] = {
        "approve": ApproveDecision,
        "reject": RejectDecision,
        "respond": RespondDecision,
    }
    if "edit" in allowed:
        by_type["edit"] = _edit_decision(name, tool)
    # Drop duplicates, keeping order: a union of one type can't take a `Discriminator`.
    members = tuple(by_type[d] for d in dict.fromkeys(allowed))
    # Type checkers can't follow a type built at runtime. At runtime, `interrupt()` checks
    # every answer against it, so what it returns is one of these decisions.
    if len(members) == 1:
        return cast("type[Decision]", members[0])
    # Check only the branch the answer's `type` names, so a bad answer gets one precise
    # error rather than one per branch.
    return cast("type[Decision]", Annotated[Union[members], Discriminator("type")])  # noqa: UP007


def _answer_message(tool_call: ToolCall, decision: RejectDecision | RespondDecision) -> ToolMessage:
    """The message the model gets in place of the tool's result."""
    if decision["type"] == "respond":
        # Skip tool execution; the human answers on behalf of the tool.
        content = decision["message"]
    elif reason := decision.get("message"):
        content = f"User rejected the tool call for `{tool_call['name']}` with reason: {reason}"
    else:
        content = (
            f"User rejected the tool call for `{tool_call['name']}` with id "
            f"{tool_call['id']}. The tool was not executed. Do not retry this tool "
            "call unless the user explicitly requests it."
        )
    return ToolMessage(
        content=content,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
        status="success" if decision["type"] == "respond" else "error",
    )


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
        config = InterruptOnConfig(
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
    """JSON schema for the args associated with the action, if edits are allowed.

    Not sent to the reviewer in either mode. In `per_call` mode the interrupt's
    `response_schema` shows the tool's own argument schema instead.
    """

    when: NotRequired[Callable[[ToolCallRequest], bool]]
    """Optional predicate controlling whether to interrupt for a given tool call.

    Receives a `ToolCallRequest` and returns `True` to interrupt or `False` to
    auto-approve. The predicate is called during `after_model` before the tool
    call is added to the batched human-in-the-loop request.

    The request is constructed with `tool=None` and a new `ToolRuntime`. The
    `ToolRuntime` copies `context`, `store`, `stream_writer`, `execution_info`,
    and `server_info` from the node-level `Runtime`, while `tool_call_id` is
    populated from the current tool call. The `tools` argument is not supplied,
    so it uses its default empty list.

    In `per_call` mode the predicate runs in `wrap_tool_call` and receives the real
    `ToolCallRequest`, with `tool` and the tool's `ToolRuntime` set.

    In both modes the predicate runs again when the run resumes, so it must return the
    same answer for the same call. In `per_call` mode, if it returns `False` on resume,
    the reviewer's answer is skipped and the tool runs, even if they rejected it.

    Example:
        ```python
        # Only interrupt delete_file calls targeting /etc
        config = InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            when=lambda req: req.tool_call["args"].get("path", "").startswith("/etc"),
        )
        ```
    """


class _HumanInTheLoopState(AgentState[ResponseT]):
    """State schema for `HumanInTheLoopMiddleware`."""

    hitl_edited_tool_calls: NotRequired[Annotated[dict[str, Action], PrivateStateAttr]]
    """Track tool call edits from `after_model`, so they can be used by `wrap_tool_call`."""


class HumanInTheLoopMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
    """Human in the loop middleware."""

    state_schema = _HumanInTheLoopState  # type: ignore[assignment]

    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        description_prefix: str = "Tool execution requires approval",
        edit_notice: str | None = _EDIT_NOTICE,
        interrupt_mode: InterruptMode = "batched",
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

                    A `when` predicate can also be provided to dynamically control
                    whether a tool call triggers an interrupt.
            description_prefix: The prefix to use when constructing action requests.

                This is used to provide context about the tool call and the action being
                requested.

                Not used if a tool has a `description` in its `InterruptOnConfig`.
            edit_notice: Text prepended to the result of a tool call a reviewer replaced
                via an `edit` decision. Pass `None` to add nothing.
            interrupt_mode: `"batched"` (default) raises one interrupt per model turn for
                all gated tool calls. `"per_call"` raises a `ToolApprovalRequest` per
                gated call, with a typed `response_schema`, each answered with a single
                `Decision` keyed by interrupt ID.
                In `per_call` mode an edit can't switch tools, its args are checked
                against the tool's argument types (validator methods on its
                `args_schema` run when the tool does), and an invalid answer raises
                `pydantic.ValidationError` without being saved. If one of several
                answers sent together is invalid, the others' tools may already have
                run even if they still show as pending, and answering again would run
                them twice: check answers against `response_schema` before sending
                them together, or send one at a time. For a
                tool whose arguments are a JSON schema rather than a Pydantic model,
                edits are shown in `response_schema` but not checked.

                In `per_call` mode, list this middleware before tool retry or
                error-handling middleware so it wraps them, and don't enable it on
                subclasses that raise their own interrupts in `after_model`. On resume,
                each paused tool call runs its middleware again up to this one, so
                middleware listed before it must not have side effects before calling
                `handler`. The tool itself runs only after the answer. As in `batched`
                mode, running the agent asynchronously (`ainvoke`, `astream`) needs
                Python 3.11 or later.

        Raises:
            ValueError: If a tool's `InterruptOnConfig` does not have a non-empty
                `allowed_decisions` list (e.g. a misspelled key or an empty list).
                An interrupt config without decisions would otherwise be silently
                dropped, disabling the approval gate for that tool.
                Also if `interrupt_mode` is invalid.
        """
        super().__init__()
        if interrupt_mode not in (modes := get_args(InterruptMode)):
            allowed = " or ".join(repr(mode) for mode in modes)
            msg = f"`interrupt_mode` must be {allowed}, got {interrupt_mode!r}."
            raise ValueError(msg)
        self.interrupt_mode = interrupt_mode
        self.edit_notice = edit_notice
        resolved_configs: dict[str, InterruptOnConfig] = {}
        for tool_name, tool_config in interrupt_on.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_configs[tool_name] = InterruptOnConfig(
                        allowed_decisions=["approve", "edit", "reject", "respond"]
                    )
            elif tool_config.get("allowed_decisions"):
                resolved_configs[tool_name] = tool_config
            else:
                msg = (
                    f"Invalid `interrupt_on` config for tool '{tool_name}': "
                    "`allowed_decisions` must be a non-empty list of decision types "
                    "(e.g. ['approve', 'reject']). Got config with keys "
                    f"{sorted(tool_config.keys())} and "
                    f"allowed_decisions={tool_config.get('allowed_decisions')!r}."
                )
                raise ValueError(msg)
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
            # Keep the model's own call in the message; `wrap_tool_call` substitutes the
            # reviewer's at execution time.
            return tool_call, None
        if decision["type"] in allowed_decisions and (
            decision["type"] == "reject" or decision["type"] == "respond"
        ):
            return tool_call, _answer_message(tool_call, decision)
        msg = (
            f"Unexpected human decision: {decision}. "
            f"Decision type '{decision.get('type')}' "
            f"is not allowed for tool '{tool_call['name']}'. "
            f"Expected one of {allowed_decisions} based on the tool's configuration."
        )
        raise ValueError(msg)

    def _should_interrupt(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> bool:
        """Return False if the `when` predicate rejects this tool call, True otherwise."""
        when = config.get("when")
        if when is None:
            return True
        runnable_config: RunnableConfig
        try:
            runnable_config = get_config()
        except RuntimeError:
            runnable_config = {}
        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            config=runnable_config,
            stream_writer=runtime.stream_writer,
            tool_call_id=tool_call["id"],
            store=runtime.store,
            execution_info=runtime.execution_info,
            server_info=runtime.server_info,
        )
        req = ToolCallRequest(
            tool_call=tool_call,
            tool=None,
            state=state,
            runtime=tool_runtime,  # type: ignore[arg-type]
        )
        return when(req)

    def _tool_approval_request(
        self, request: ToolCallRequest, config: InterruptOnConfig
    ) -> ToolApprovalRequest:
        """Build the per-call interrupt value for one gated tool call.

        Raises:
            ValueError: If the tool call has no ID. It can't run without one, so the
                reviewer isn't asked to approve it.
        """
        tool_call = request.tool_call
        tool_call_id = tool_call["id"]
        # Only `None`: an empty ID still runs, so per-call keeps working where batched does.
        if tool_call_id is None:
            msg = (
                f"Tool call `{tool_call['name']}` has no ID, so its result can't be matched "
                "to it. Make sure the chat model returns tool call IDs."
            )
            raise ValueError(msg)
        # A description factory gets the graph `Runtime`, as in batched mode.
        runtime: Runtime[ContextT] = get_runtime()
        action_request, _ = self._create_action_and_config(
            tool_call, config, request.state, runtime
        )
        return ToolApprovalRequest(
            type="tool_approval",
            tool_call_id=tool_call_id,
            name=tool_call["name"],
            args=tool_call["args"],
            description=action_request.get("description", ""),
        )

    def _resolve_per_call(
        self, request: ToolCallRequest
    ) -> tuple[ToolCallRequest, Action | None] | ToolMessage:
        """Ask the reviewer about one tool call, if it's gated.

        Returns the request to run, with the reviewer's call if they edited it, or the
        message to return instead of running the tool.
        """
        tool_call = request.tool_call
        config = self.interrupt_on.get(tool_call["name"])
        if config is None:
            return request, None
        when = config.get("when")
        if when is not None and not when(request):
            return request, None

        value = self._tool_approval_request(request, config)
        response_schema = _decision_schema(
            config["allowed_decisions"], tool_call["name"], request.tool
        )
        # LangGraph parses the answer against `response_schema` before saving it, so it
        # comes back as one of this tool's allowed decisions.
        decision = interrupt(value, response_schema=response_schema)
        if decision["type"] == "approve":
            return request, None
        if decision["type"] == "edit":
            # The schema pinned the tool name, so this is always the same tool.
            executed = decision["edited_action"]
            return self._apply_edit(request, executed), executed
        return _answer_message(tool_call, decision)

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
        if self.interrupt_mode == "per_call":
            # Interrupts are raised per tool call, in `wrap_tool_call`.
            return None
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
                if not self._should_interrupt(tool_call, config, state, runtime):
                    continue
                action_request, review_config = self._create_action_and_config(
                    tool_call, config, state, runtime
                )
                action_requests.append(action_request)
                review_configs.append(review_config)
                interrupt_indices.append(idx)

        # If no interrupts needed, return early, dropping any earlier turn's edits so
        # they cannot be applied to this turn's tool calls.
        if not action_requests:
            if state.get(_EDITED_TOOL_CALLS_KEY):
                return {_EDITED_TOOL_CALLS_KEY: {}}
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
        edited_tool_calls: dict[str, Action] = {}
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
                    if decision["type"] == "edit" and (edited_id := revised_tool_call.get("id")):
                        edited_tool_calls[edited_id] = decision["edited_action"]
                if tool_message:
                    artificial_tool_messages.append(tool_message)
            else:
                # This was auto-approved - keep original
                revised_tool_calls.append(tool_call)

        # Update the AI message to only include approved tool calls
        last_ai_msg.tool_calls = revised_tool_calls

        # `wrap_tool_call` reads this back to substitute and annotate the call. Always
        # written, so an earlier turn's edits cannot survive into this one.
        return {
            "messages": [last_ai_msg, *artificial_tool_messages],
            _EDITED_TOOL_CALLS_KEY: edited_tool_calls,
        }

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

    def _reviewer_edit(self, request: ToolCallRequest) -> Action | None:
        """The reviewer's replacement for this call, if an `edit` decision replaced it."""
        tool_call_id = request.tool_call.get("id")
        if not tool_call_id:
            return None
        edited = request.state.get(_EDITED_TOOL_CALLS_KEY) or {}
        if tool_call_id in edited:
            executed: Action = edited[tool_call_id]
            return executed
        return None

    def _apply_edit(self, request: ToolCallRequest, executed: Action) -> ToolCallRequest:
        """Point the request at the reviewer's call, resolving a redirected tool.

        Raises:
            ValueError: If the reviewer named a tool the agent does not have.
        """
        tool_call: ToolCall = {
            **request.tool_call,
            "name": executed["name"],
            "args": executed["args"],
        }
        if executed["name"] == request.tool_call["name"]:
            return request.override(tool_call=tool_call)

        # `tool_call["name"]` and `tool` must stay in agreement.
        available = request.runtime.tools
        tool = next((t for t in available if t.name == executed["name"]), None)
        if tool is None:
            names = ", ".join(sorted(t.name for t in available))
            msg = (
                f"Reviewer edited tool call {request.tool_call['id']!r} to "
                f"{executed['name']!r}, which is not an available tool. "
                f"Available tools: {names}."
            )
            raise ValueError(msg)
        return request.override(tool_call=tool_call, tool=tool)

    def _notice(self, executed: Action, *, has_content: bool) -> str:
        """The notice text, stating the call that actually ran."""
        notice = (
            f"{self.edit_notice} Executed instead: {executed['name']} with arguments "
            f"{json.dumps(executed['args'], default=str)}."
        )
        return f"{notice}\n\nTool response:" if has_content else notice

    def _prepend_notice(self, message: ToolMessage, executed: Action) -> ToolMessage:
        """Return `message` with the reviewer-edit notice prepended to its content."""
        if not self.edit_notice:
            return message
        edit_notice = self._notice(executed, has_content=bool(message.content))

        content: str | list[str | dict[Any, Any]]
        if isinstance(message.content, str):
            if edit_notice in message.content:
                return message
            separator = "\n" if message.content else ""
            content = f"{edit_notice}{separator}{message.content}"
        else:
            if any(edit_notice in str(block) for block in message.content):
                return message
            # Match the surrounding block shape; providers may reject mixed lists.
            notice: str | dict[Any, Any] = (
                edit_notice
                if message.content and all(isinstance(b, str) for b in message.content)
                else {"type": "text", "text": edit_notice}
            )
            content = [notice, *message.content]

        return message.model_copy(update={"content": content})

    def _annotate_edited_result(
        self,
        result: ToolMessage | Command[Any],
        request: ToolCallRequest,
        executed: Action | None,
    ) -> ToolMessage | Command[Any]:
        """Tell the model a reviewer replaced the call, and with what."""
        if not self.edit_notice or executed is None:
            return result

        if isinstance(result, ToolMessage):
            return self._prepend_notice(result, executed)

        # A `Command` carries the `ToolMessage` in its state update.
        if not isinstance(result, Command) or not isinstance(result.update, dict):
            return result
        messages = result.update.get("messages")
        if not isinstance(messages, list):
            return result
        tool_call_id = request.tool_call.get("id")
        return replace(
            result,
            update={
                **result.update,
                "messages": [
                    self._prepend_notice(message, executed)
                    if isinstance(message, ToolMessage) and message.tool_call_id == tool_call_id
                    else message
                    for message in messages
                ],
            },
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Prepend reviewer-edit guidance to the result of an edited tool call.

        In `per_call` mode this is also where the interrupt is raised.

        Args:
            request: The tool call request being executed.
            handler: Callable that executes the tool.

        Returns:
            The tool result, with a note prepended when a reviewer edited the call.

        Raises:
            ValidationError: In `per_call` mode, if the reviewer's answer doesn't match
                the tool call's `response_schema`. Nothing is saved, so the call can be
                answered again.
            ValueError: In `per_call` mode, if a gated tool call has no ID.
        """
        if self.interrupt_mode == "per_call":
            resolved = self._resolve_per_call(request)
            if isinstance(resolved, ToolMessage):
                return resolved
            to_run, executed = resolved
            return self._annotate_edited_result(handler(to_run), to_run, executed)
        executed = self._reviewer_edit(request)
        if executed is not None:
            request = self._apply_edit(request, executed)
        return self._annotate_edited_result(handler(request), request, executed)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of `wrap_tool_call`.

        In `per_call` mode this is also where the interrupt is raised.

        Args:
            request: The tool call request being executed.
            handler: Awaitable callable that executes the tool.

        Returns:
            The tool result, with a note prepended when a reviewer edited the call.

        Raises:
            ValidationError: In `per_call` mode, if the reviewer's answer doesn't match
                the tool call's `response_schema`. Nothing is saved, so the call can be
                answered again.
            ValueError: In `per_call` mode, if a gated tool call has no ID.
        """
        if self.interrupt_mode == "per_call":
            resolved = self._resolve_per_call(request)
            if isinstance(resolved, ToolMessage):
                return resolved
            to_run, executed = resolved
            return self._annotate_edited_result(await handler(to_run), to_run, executed)
        executed = self._reviewer_edit(request)
        if executed is not None:
            request = self._apply_edit(request, executed)
        return self._annotate_edited_result(await handler(request), request, executed)

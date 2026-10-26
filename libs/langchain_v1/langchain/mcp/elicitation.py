"""Answer MCP elicitation requests with a LangGraph interrupt.

A server needing input mid-call returns an `InputRequiredResult` and expects the
`tools/call` to be retried with answers. This module drives that loop, sourcing
each answer from `interrupt()` so the human reviewing the agent answers too.

We drive the loop here rather than via the SDK's `run_input_required_driver`
because that driver answers from callbacks run concurrently in a task group:
LangGraph matches resume values to `interrupt()` calls by order, so concurrent
firing scrambles the matching, and FastMCP would swallow the `GraphInterrupt` as
an MCP error. Calling `interrupt()` from this frame keeps one per round.

Only elicitation is answered. Sampling, roots, and continuation rounds are
refused rather than half-served.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypedDict, TypeVar

from fastmcp.client.group import ClientGroup
from langgraph.types import interrupt
from mcp.types import (
    CallToolResult,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    InputResponses,
)
from mcp.types.version import LATEST_MODERN_VERSION, is_version_at_least
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from fastmcp.client import Client
    from mcp.client.session import ClientSession
    from mcp.types import InputRequest


_ResultT = TypeVar("_ResultT")


ELICITATION_INTERRUPT_TYPE: Final = "mcp_elicitation"
"""Discriminator on the interrupt payload, so a handler can recognize it."""

_ACTIONS: Final = ("accept", "decline", "cancel")
"""Actions the wire accepts, for validating a resumed answer."""

MCPFormContent: TypeAlias = dict[str, str | int | float | bool | list[str] | None]
"""Values a form answer may carry, as narrow as `mcp.types.ElicitResult.content`."""


class MCPElicitationFormRequest(TypedDict):
    """A request for data matching a schema.

    Attributes:
        key: The server's identifier for this request. An answer must be
            returned under the same key.
        message: The server's prompt, intended for a human to read.
        mode: Always `'form'`.
        requested_schema: JSON schema the answer's `content` must satisfy.
    """

    key: str
    message: str
    mode: Literal["form"]
    requested_schema: dict[str, Any]


class MCPElicitationUrlRequest(TypedDict):
    """A request for the human to visit an address.

    Attributes:
        key: The server's identifier for this request. An answer must be
            returned under the same key.
        message: The server's prompt, intended for a human to read.
        mode: Always `'url'`.
        url: The address the human should visit.
    """

    key: str
    message: str
    mode: Literal["url"]
    url: str


MCPElicitationRequest: TypeAlias = MCPElicitationFormRequest | MCPElicitationUrlRequest
"""One question an MCP server asks before finishing a tool call. Narrow on `mode`."""


class MCPElicitationInterrupt(TypedDict):
    """Interrupt payload raised while an MCP tool call waits on input.

    Attributes:
        type: Always `'mcp_elicitation'`.
        tool_name: The MCP tool whose call is waiting.
        requests: Every question this round, in the order they should be asked.
    """

    type: Literal["mcp_elicitation"]
    tool_name: str
    requests: list[MCPElicitationRequest]


class MCPElicitationAccept(TypedDict):
    """An answer to an `MCPElicitationRequest`.

    Attributes:
        action: Always `'accept'`.
        content: The answer itself, matching the request's `requested_schema`.
            Expected for a `'form'` request; a `'url'` request carries none.
    """

    action: Literal["accept"]
    content: NotRequired[MCPFormContent | None]


class MCPElicitationDecline(TypedDict):
    """A refusal to answer an `MCPElicitationRequest`, leaving the call to go on.

    Attributes:
        action: Always `'decline'`.
    """

    action: Literal["decline"]


class MCPElicitationCancel(TypedDict):
    """A refusal that abandons the tool call rather than just the question.

    Attributes:
        action: Always `'cancel'`.
    """

    action: Literal["cancel"]


MCPElicitationResponse: TypeAlias = (
    MCPElicitationAccept | MCPElicitationDecline | MCPElicitationCancel
)
"""One answer to an `MCPElicitationRequest`. Narrow on `action`; only accept carries content."""


class MCPElicitationResume(TypedDict):
    """Value used to resume a run interrupted by an MCP elicitation.

    Attributes:
        responses: One answer per request, keyed by the request's `key`.
    """

    responses: dict[str, MCPElicitationResponse]


def _describe_request(key: str, request: ElicitRequest) -> MCPElicitationRequest:
    """Render one server request as a payload a human can act on."""
    params = request.params
    if isinstance(params, ElicitRequestFormParams):
        return MCPElicitationFormRequest(
            key=key,
            message=params.message,
            mode="form",
            requested_schema=params.requested_schema,
        )
    return MCPElicitationUrlRequest(
        key=key,
        message=params.message,
        mode="url",
        url=str(params.url),
    )


def _elicit_requests(
    requests: dict[str, InputRequest],
    tool_name: str,
) -> dict[str, ElicitRequest]:
    """Narrow a round's requests to elicitations, raising on sampling or roots."""
    unsupported = sorted(
        f"{key} ({request.method})"
        for key, request in requests.items()
        if not isinstance(request, ElicitRequest)
    )
    if unsupported:
        msg = (
            f"The MCP tool {tool_name!r} asked for input that interrupt-based "
            f"elicitation cannot answer: {', '.join(unsupported)}. Only "
            "elicitation requests are supported."
        )
        raise NotImplementedError(msg)
    return {key: request for key, request in requests.items() if isinstance(request, ElicitRequest)}


def _build_responses(
    requests: dict[str, ElicitRequest],
    answers: dict[str, MCPElicitationResponse],
    tool_name: str,
) -> InputResponses:
    """Turn resumed answers into the server's responses, raising on any missing or malformed."""
    missing = sorted(set(requests) - set(answers))
    if missing:
        msg = (
            f"Resuming the MCP tool {tool_name!r} needs an answer for every "
            f"elicitation request, but these had none: {', '.join(missing)}."
        )
        raise ValueError(msg)

    responses: InputResponses = {}
    for key in requests:
        answer = answers[key]
        action = answer.get("action")
        if action not in _ACTIONS:
            msg = (
                f"Elicitation answer for {key!r} has action {action!r}; expected "
                "'accept', 'decline', or 'cancel'."
            )
            raise ValueError(msg)
        content = answer.get("content") if action == "accept" else None
        responses[key] = ElicitResult(action=action, content=content)
    return responses


_ARMED_MARKER: Final = "_langchain_mcp_interrupt_armed"
"""Attribute stamped on a client the adapter armed for interrupt elicitation."""


async def _declare_elicitation_capability(*_: object) -> dict[str, Any]:
    """Elicitation handler that only exists to advertise the capability.

    FastMCP advertises `elicitation` only when a handler is set, so a client
    that should drive the interrupt loop needs one. The loop answers from the
    tool call, not this handler, so it never runs on a modern server. It only
    fires when a legacy server initiates elicitation the old way — which the
    interrupt loop cannot answer, hence the error.
    """
    msg = (
        "This MCP server asked for input over the legacy server-initiated "
        "elicitation path, which interrupt-based elicitation cannot answer. It "
        "needs a server that returns its requests as an `InputRequiredResult`, "
        "or a pre-built client with your own `elicitation_handler`."
    )
    raise NotImplementedError(msg)


def _arm_for_interrupts(client: Client[Any]) -> None:
    """Set the sentinel handler and mark the client as adapter-armed."""
    client.set_elicitation_callback(_declare_elicitation_capability)
    setattr(client, _ARMED_MARKER, True)


def _drives_interrupts(client: Client[Any]) -> bool:
    """Report whether a call through `client` should drive the interrupt loop.

    True only for a client the adapter armed (not one carrying a caller's own
    handler) that is connected to a modern-era server. The `InputRequiredResult`
    the loop answers is a modern feature (SEP-2322); a legacy server never
    returns one, so the call falls back to a plain `call_tool` there.
    """
    if not getattr(client, _ARMED_MARKER, False):
        return False
    version = getattr(client, "protocol_version", None)
    return version is not None and is_version_at_least(version, LATEST_MODERN_VERSION)


async def _await_monitored(client: Client[Any], coro: Coroutine[Any, Any, _ResultT]) -> _ResultT:
    """Race a session request against the session task so an error can't hang it.

    On HTTP transports a server error surfaces in the background session task,
    not in the awaiting coroutine. FastMCP's guard for this is private, so warn
    and fall back if it ever moves.
    """
    monitored = getattr(client, "_await_with_session_monitoring", None)
    if monitored is None:
        warnings.warn(
            "This version of FastMCP does not expose "
            "`Client._await_with_session_monitoring`, so an MCP elicitation "
            "round cannot be raced against the session task. A transport "
            "failure mid-elicitation may hang instead of raising.",
            RuntimeWarning,
            stacklevel=2,
        )
        return await coro
    result: _ResultT = await monitored(coro)
    return result


async def _resolve_session(
    client: Client[Any] | ClientGroup, tool_name: str
) -> tuple[Client[Any], ClientSession, str]:
    """Resolve the member client, its session, and the server's name for a tool.

    For a group, `list_tools` namespaces the catalog as `{server}_{tool}` but the
    serving client only knows the tool by its own name; `resolve_tool` maps the
    namespaced name back to that member and its upstream name. For a single
    client the tool is already addressed by its own name.

    Args:
        client: The armed client or group serving the tool.
        tool_name: The tool's name as this adapter published it.

    Returns:
        The member client, its connected session, and the name the member's
        server knows the tool by.
    """
    if isinstance(client, ClientGroup):
        route = await client.resolve_tool(tool_name)
        return route.client, route.client.session, route.upstream_name
    return client, client.session, tool_name


async def _call_tool_with_interrupts(
    client: Client[Any] | ClientGroup,
    tool_name: str,
    arguments: dict[str, Any],
) -> CallToolResult:
    """Call an MCP tool, answering each round of requested input with an interrupt.

    `interrupt()` unwinds the whole call, so on resume it is re-issued from the
    first round. A server that asks before doing work repeats nothing; one that
    works first repeats that work once per round.

    Args:
        client: A connected client armed to advertise the elicitation capability.
        tool_name: The MCP tool to call.
        arguments: Arguments for the tool.

    Returns:
        The tool's terminal result.

    Raises:
        GraphInterrupt: For each unanswered round. This is the mechanism.
        NotImplementedError: On a sampling/roots request or a continuation round.
        ValueError: If a resumed answer is missing or malformed.
    """
    # FastMCP's public `call_tool` no longer forwards `allow_input_required`; it
    # drives the input-required loop itself via a concurrent driver, which is the
    # exact behavior this module replaces with one `interrupt()` per round. So
    # drive it against the member session, whose `call_tool` still returns the
    # `InputRequiredResult` for us to answer.
    member, session, upstream_name = await _resolve_session(client, tool_name)

    result = await _await_monitored(
        member, session.call_tool(upstream_name, arguments, allow_input_required=True)
    )

    while isinstance(result, InputRequiredResult):
        if not result.input_requests:
            # A round with only `request_state` is a continuation: the server is
            # still working, with no question to interrupt on. Refused like sampling.
            msg = (
                f"The MCP tool {tool_name!r} returned a continuation round with "
                "no input requests. Interrupt-based elicitation only answers "
                "elicitation requests, not long-running work that resumes over "
                "several round trips."
            )
            raise NotImplementedError(msg)

        requests = _elicit_requests(result.input_requests, tool_name)
        request: MCPElicitationInterrupt = {
            "type": ELICITATION_INTERRUPT_TYPE,
            "tool_name": tool_name,
            "requests": [_describe_request(key, req) for key, req in requests.items()],
        }
        resume: MCPElicitationResume = interrupt(request)
        responses = _build_responses(requests, resume["responses"], tool_name)

        result = await _await_monitored(
            member,
            session.call_tool(
                upstream_name,
                arguments,
                input_responses=responses,
                request_state=result.request_state,  # opaque; echoed back verbatim
                allow_input_required=True,
            ),
        )

    return result


__all__ = [
    "ELICITATION_INTERRUPT_TYPE",
    "MCPElicitationAccept",
    "MCPElicitationCancel",
    "MCPElicitationDecline",
    "MCPElicitationFormRequest",
    "MCPElicitationInterrupt",
    "MCPElicitationRequest",
    "MCPElicitationResponse",
    "MCPElicitationResume",
    "MCPElicitationUrlRequest",
    "MCPFormContent",
]

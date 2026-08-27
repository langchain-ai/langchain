"""Answer MCP elicitation requests with a LangGraph interrupt.

A server that needs input mid-call answers `tools/call` with an
`InputRequiredResult` carrying the requests it wants fulfilled, and expects the
call to be retried with the answers. This module drives that loop and sources
each answer from `interrupt()`, so the human already reviewing an agent's work
answers the server's question too.

The loop is driven here rather than through the SDK's own
`run_input_required_driver` because that driver answers each request from a
callback, run concurrently in a task group. LangGraph matches resume values to
`interrupt()` calls by their order in the node, so firing them concurrently
would scramble that matching — and FastMCP converts any exception a callback
raises into an MCP error, swallowing the `GraphInterrupt` that suspends the
graph. Calling `interrupt()` from this module's own frame keeps one interrupt
per round and lets it propagate. The retry bounds mirror the SDK driver's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, TypedDict, TypeVar

import anyio
from langgraph.types import interrupt
from mcp import InputRequiredRoundsExceededError
from mcp.types import (
    CallToolResult,
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    InputResponses,
)
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from fastmcp.client import Client
    from mcp.types import InputRequest


_ResultT = TypeVar("_ResultT")


_STATE_ONLY_BACKOFF_INITIAL_SECONDS = 0.05
_STATE_ONLY_BACKOFF_CAP_SECONDS = 0.25
"""Backoff for rounds that ask nothing, matching the SDK's own driver."""

ELICITATION_INTERRUPT_TYPE: Final = "mcp_elicitation"
"""Discriminator on the interrupt payload, so a handler can recognize it."""

_ACTIONS: Final = ("accept", "decline", "cancel")
"""Actions the wire accepts, for validating a resumed answer."""

MCPFormContent: TypeAlias = dict[str, str | int | float | bool | list[str] | None]
"""Values a form answer may carry.

As narrow as `mcp.types.ElicitResult.content`, which validates the answer on its
way to the server — a wider annotation would promise more than the wire takes.
"""


class MCPElicitationRequest(TypedDict):
    """One question an MCP server is asking before it can finish a tool call.

    Attributes:
        key: The server's identifier for this request. An answer must be
            returned under the same key.
        message: The server's prompt, intended for a human to read.
        mode: `'form'` when the server wants data matching `requested_schema`,
            `'url'` when it wants the human to visit `url`.
        requested_schema: JSON schema the answer's `content` must satisfy, for
            a `'form'` request.
        url: The address the human should visit, for a `'url'` request.
    """

    key: str
    message: str
    mode: Literal["form", "url"]
    requested_schema: NotRequired[dict[str, Any]]
    url: NotRequired[str]


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


class MCPElicitationRefusal(TypedDict):
    """A refusal to answer an `MCPElicitationRequest`.

    Attributes:
        action: `'decline'` to refuse the question, or `'cancel'` to abandon the
            tool call.
    """

    action: Literal["decline", "cancel"]


MCPElicitationResponse: TypeAlias = MCPElicitationAccept | MCPElicitationRefusal
"""One answer to an `MCPElicitationRequest`.

Split so that only an accept can carry content, matching what is actually sent:
content on a refusal is dropped rather than forwarded.
"""


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
        return MCPElicitationRequest(
            key=key,
            message=params.message,
            mode="form",
            requested_schema=params.requested_schema,
        )
    return MCPElicitationRequest(
        key=key,
        message=params.message,
        mode="url",
        url=str(params.url),
    )


def _elicit_requests(
    requests: dict[str, InputRequest],
    tool_name: str,
) -> dict[str, ElicitRequest]:
    """Narrow a round's requests to elicitations, rejecting what is unsupported.

    Answering sampling or roots is FastMCP's job, and driving this loop by hand
    bypasses the callbacks that would do it, so those raise instead.
    """
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
    """Turn resumed answers into the responses the server is expecting.

    Every request needs an answer under its own key; a missing or malformed one
    raises rather than being silently dropped from the reply.
    """
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


async def _declare_elicitation_capability(*_: object) -> dict[str, Any]:
    """Stand in for an elicitation handler so the client advertises the capability.

    FastMCP declares `elicitation` only when the callback differs by identity
    from its own default. Interrupt-driven elicitation answers from the tool
    call rather than from a callback, so this exists only to make that
    declaration — running it means a server used the legacy server-initiated
    path instead, which an interrupt cannot answer.
    """
    msg = (
        "This MCP server asked for input over the legacy server-initiated "
        "path, which `elicitation='interrupt'` cannot answer. Interrupt-based "
        "elicitation needs a server that returns its input requests as an "
        "`InputRequiredResult`."
    )
    raise NotImplementedError(msg)


async def _await_monitored(client: Client[Any], coro: Coroutine[Any, Any, _ResultT]) -> _ResultT:
    """Await a session request so a dying session cannot leave it hanging.

    On HTTP transports a server error surfaces in the background session task,
    not in the coroutine awaiting the reply, so `fastmcp.Client` races the two.
    Driving this loop by hand means reaching for the same guard, which FastMCP
    exposes only privately — hence the fallback if the helper ever moves.
    """
    monitored = getattr(client, "_await_with_session_monitoring", None)
    if monitored is None:
        return await coro
    result: _ResultT = await monitored(coro)
    return result


async def _call_tool_with_interrupts(
    client: Client[Any],
    tool_name: str,
    arguments: dict[str, Any],
) -> CallToolResult:
    """Call an MCP tool, answering any input it asks for with an interrupt.

    Each round of requests becomes one `interrupt()`, and the answers are sent
    back on a retry echoing the server's opaque `request_state`.

    Because `interrupt()` unwinds the whole tool call, the call is re-issued
    from its first round when the run resumes. A server that asks before doing
    any work — the shape the protocol is designed around — repeats nothing. One
    that works first and asks later repeats that work once per round.

    Args:
        client: A connected FastMCP client, built with an `elicitation_handler`
            so it advertises the capability, or servers will refuse to ask.
        tool_name: The MCP tool to call.
        arguments: Arguments for the tool.

    Returns:
        The tool's terminal result.

    Raises:
        GraphInterrupt: Every time the server asks something that has not been
            answered yet. This is the mechanism, not a failure.
        InputRequiredRoundsExceededError: If the server keeps asking past
            `client.input_required_max_rounds`, so a server that never reaches
            a terminal result cannot loop forever.
        NotImplementedError: If the server asks for sampling or roots.
        ValueError: If a resumed answer is missing or malformed.
    """
    session = client.session
    max_rounds = client.input_required_max_rounds
    result = await _await_monitored(
        client, session.call_tool(tool_name, arguments, allow_input_required=True)
    )

    rounds = 0
    state_only_delay = _STATE_ONLY_BACKOFF_INITIAL_SECONDS
    while isinstance(result, InputRequiredResult):
        rounds += 1
        if rounds > max_rounds:
            raise InputRequiredRoundsExceededError(max_rounds)

        responses: InputResponses | None = None
        if result.input_requests:
            state_only_delay = _STATE_ONLY_BACKOFF_INITIAL_SECONDS
            requests = _elicit_requests(result.input_requests, tool_name)
            request: MCPElicitationInterrupt = {
                "type": ELICITATION_INTERRUPT_TYPE,
                "tool_name": tool_name,
                "requests": [_describe_request(key, req) for key, req in requests.items()],
            }
            resume: MCPElicitationResume = interrupt(request)
            responses = _build_responses(requests, resume["responses"], tool_name)
        else:
            # A round carrying only `request_state` means the server is still
            # working and wants to be asked again. Nobody needs to be
            # interrupted for that; just back off so the retry is not a spin.
            await anyio.sleep(state_only_delay)
            state_only_delay = min(state_only_delay * 2, _STATE_ONLY_BACKOFF_CAP_SECONDS)

        result = await _await_monitored(
            client,
            session.call_tool(
                tool_name,
                arguments,
                input_responses=responses,
                # Opaque to us: echoed back byte-exact, never inspected.
                request_state=result.request_state,
                allow_input_required=True,
            ),
        )

    return result


__all__ = [
    "ELICITATION_INTERRUPT_TYPE",
    "MCPElicitationAccept",
    "MCPElicitationInterrupt",
    "MCPElicitationRefusal",
    "MCPElicitationRequest",
    "MCPElicitationResponse",
    "MCPElicitationResume",
    "MCPFormContent",
]

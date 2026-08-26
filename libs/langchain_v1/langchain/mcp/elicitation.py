"""Answer MCP elicitation requests with a LangGraph interrupt.

A server that needs input mid-call answers `tools/call` with an
`InputRequiredResult` carrying the requests it wants fulfilled, and expects the
call to be retried with the answers. This module drives that loop and sources
each answer from `interrupt()`, so the human already reviewing an agent's work
answers the server's question too.

The loop is driven here rather than through FastMCP's own handler because
FastMCP converts any exception a handler raises into an MCP error, which would
swallow the `GraphInterrupt` that suspends the graph. Calling `interrupt()` from
this module's own frame lets it propagate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

import anyio
from langgraph.types import interrupt
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
    from fastmcp.client import Client
    from mcp.types import InputRequest


_STILL_WORKING_SLEEP_SECONDS = 0.05
"""Pause before retrying a round that asked nothing, so the retry is not a spin."""

ELICITATION_INTERRUPT_TYPE = "mcp_elicitation"
"""Discriminator on the interrupt payload, so a handler can recognize it."""


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


class MCPElicitationResponse(TypedDict):
    """One answer to an `MCPElicitationRequest`.

    Attributes:
        action: `'accept'` to answer, `'decline'` to refuse the question, or
            `'cancel'` to abandon the tool call.
        content: The answer itself, matching the request's `requested_schema`.
            Required when accepting a `'form'` request.
    """

    action: Literal["accept", "decline", "cancel"]
    content: NotRequired[dict[str, Any] | None]


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

    Args:
        requests: Every input request the server embedded in this round.
        tool_name: The tool being called, for the error message.

    Returns:
        The elicitation requests, keyed as the server keyed them.

    Raises:
        NotImplementedError: If the server asked for sampling or roots.
            Answering those is FastMCP's job, and driving this loop by hand
            bypasses the callbacks that would do it.
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

    Args:
        requests: The elicitation requests this round.
        answers: Answers keyed by request key, as resumed from the interrupt.
        tool_name: The tool being called, for the error message.

    Returns:
        One `ElicitResult` per request, under the server's own keys.

    Raises:
        ValueError: If an answer is missing, or carries an unknown action.
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
        if action not in ("accept", "decline", "cancel"):
            msg = (
                f"Elicitation answer for {key!r} has action {action!r}; expected "
                "'accept', 'decline', or 'cancel'."
            )
            raise ValueError(msg)
        content = answer.get("content") if action == "accept" else None
        responses[key] = ElicitResult(action=action, content=content)
    return responses


async def _call_tool_with_interrupts(
    client: Client[Any],
    tool_name: str,
    arguments: dict[str, Any],
) -> CallToolResult:
    """Call an MCP tool, answering any input it asks for with an interrupt.

    Drives the server's multi-round-trip loop directly: each round of requests
    becomes one `interrupt()`, and the answers are sent back on a retry that
    echoes the server's opaque `request_state`.

    Because `interrupt()` unwinds the whole tool call, the call is re-issued
    from its first round when the run resumes. A server that asks for input
    before doing any work — the shape the protocol is designed around — repeats
    nothing. A server that works first and asks later repeats that work once per
    round of questions.

    Args:
        client: A connected FastMCP client. It must have been built with an
            `elicitation_handler` so it advertises the elicitation capability,
            or servers will refuse to ask.
        tool_name: The MCP tool to call.
        arguments: Arguments for the tool.

    Returns:
        The tool's terminal result.

    Raises:
        GraphInterrupt: Every time the server asks something that has not been
            answered yet. This is the mechanism, not a failure.
        NotImplementedError: If the server asks for sampling or roots.
        ValueError: If a resumed answer is missing or malformed.
    """
    session = client.session
    result = await session.call_tool(tool_name, arguments, allow_input_required=True)

    while isinstance(result, InputRequiredResult):
        responses: InputResponses | None = None
        if result.input_requests:
            requests = _elicit_requests(result.input_requests, tool_name)
            request: MCPElicitationInterrupt = {
                "type": "mcp_elicitation",
                "tool_name": tool_name,
                "requests": [_describe_request(key, req) for key, req in requests.items()],
            }
            answers = interrupt(request)["responses"]
            responses = _build_responses(requests, answers, tool_name)
        else:
            # A round carrying only `request_state` means the server is still
            # working and wants to be asked again. Nobody needs to be
            # interrupted for that; just pause so the retry is not a spin.
            await anyio.sleep(_STILL_WORKING_SLEEP_SECONDS)

        result = await session.call_tool(
            tool_name,
            arguments,
            input_responses=responses,
            # Opaque to us: echoed back byte-exact, never inspected.
            request_state=result.request_state,
            allow_input_required=True,
        )

    return result


__all__ = [
    "ELICITATION_INTERRUPT_TYPE",
    "MCPElicitationInterrupt",
    "MCPElicitationRequest",
    "MCPElicitationResponse",
    "MCPElicitationResume",
]

"""Bridge MCP elicitation to LangGraph interrupts (2026-07-28 input-required flow).

On the modern MCP protocol there is no server->client back-channel, so elicitation is
delivered as an `InputRequiredResult` (SEP-2322): a tool call returns a request for input,
the client answers it, and the call is retried with the answer. We drive that loop here
and surface each elicitation as a LangGraph `interrupt()`, so the human answers by resuming
the graph.

Only servers speaking the modern input-required mechanism trigger this path: older servers
never return an `InputRequiredResult`, so a plain call runs unchanged. The side-effectful
part of a guard-pattern tool runs exactly once; only the (side-effect-free) input check
re-runs on resume, per the guard-pattern contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import ToolException
from langgraph.types import interrupt
from mcp.types import ElicitRequest, ElicitResult, InputRequiredResult
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from fastmcp import Client
    from fastmcp.client.transports import ClientTransport
    from mcp.types import CallToolResult, InputResponses

_VALID_ACTIONS = ("accept", "decline", "cancel")


class MCPElicitation(TypedDict):
    """Value surfaced by `interrupt()` when an MCP server elicits input.

    Resume the graph with `Command(resume={"action": ..., "content": ...})`, where `action`
    is one of `"accept"`, `"decline"`, or `"cancel"`, and `content` (on accept) matches
    `response_schema`.
    """

    type: Literal["mcp_elicitation"]
    message: str
    mode: str
    response_schema: dict[str, Any]
    url: NotRequired[str]


async def call_tool_with_elicitation(
    client: Client[ClientTransport],
    name: str,
    arguments: dict[str, Any],
) -> CallToolResult:
    """Call an MCP tool, bridging any modern-spec elicitation to `interrupt()`.

    Args:
        client: Connected FastMCP client.
        name: Tool name.
        arguments: Tool arguments.

    Returns:
        The terminal tool result once all elicitation rounds are answered.

    Raises:
        ToolException: If the server requests unsupported (non-elicitation) input.
    """
    session = client.session
    result = await session.call_tool(name, arguments, allow_input_required=True)

    while isinstance(result, InputRequiredResult):
        responses: InputResponses = {}
        for key, request in (result.input_requests or {}).items():
            if not isinstance(request, ElicitRequest):
                msg = (
                    f"MCP tool {name!r} requested unsupported input "
                    f"{type(request).__name__!r}; only elicitation is supported."
                )
                raise ToolException(msg)
            responses[key] = _resolve_elicitation(request)

        result = await session.call_tool(
            name,
            arguments,
            input_responses=responses,
            request_state=result.request_state,
            allow_input_required=True,
        )

    return result


def _resolve_elicitation(request: ElicitRequest) -> ElicitResult:
    """Surface one elicitation request as an interrupt and map the resume to a result."""
    params = request.params
    payload: MCPElicitation = {
        "type": "mcp_elicitation",
        "mode": getattr(params, "mode", "form"),
        "message": params.message,
        "response_schema": getattr(params, "requested_schema", {}),
    }
    url = getattr(params, "url", None)
    if url is not None:
        payload["url"] = str(url)

    answer = interrupt(payload)
    if not isinstance(answer, dict) or answer.get("action") not in _VALID_ACTIONS:
        msg = (
            f"Invalid elicitation response {answer!r}; resume with "
            f'{{"action": one of {_VALID_ACTIONS}, "content": ...}}.'
        )
        raise ToolException(msg)

    action = answer["action"]
    content = answer.get("content") if action == "accept" else None
    return ElicitResult(action=action, content=content)

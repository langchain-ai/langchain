"""Bridge MCP elicitation callbacks to LangGraph interrupts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from langchain_core.tools import ToolException
from langgraph.types import interrupt
from mcp.types import ElicitRequestParams as _ElicitRequestParams
from mcp.types import ElicitResult as _ElicitResult
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain.mcp.callbacks import CallbackContext

ElicitationMode: TypeAlias = Literal["interrupt"]
"""How MCP tools surface server-initiated elicitation requests."""


class MCPFormElicitation(TypedDict):
    """A structured form an MCP server asks the application to present."""

    type: Literal["mcp_elicitation"]
    mode: Literal["form"]
    server: str
    message: str
    response_schema: dict[str, Any]


class MCPUrlElicitation(TypedDict):
    """A URL an MCP server asks the application to present for consent."""

    type: Literal["mcp_elicitation"]
    mode: Literal["url"]
    server: str
    message: str
    url: str


MCPElicitation: TypeAlias = MCPFormElicitation | MCPUrlElicitation
"""A discriminated MCP elicitation interrupt payload."""


MCPFormValue: TypeAlias = str | int | float | bool | list[str] | None
"""A primitive value permitted by MCP form-mode elicitation."""

MCPFormContent: TypeAlias = dict[str, MCPFormValue]
"""The flat object content accepted by an MCP form-mode elicitation request."""


class MCPFormElicitationAcceptedResponse(TypedDict):
    """An accepted form response with values matching the response schema."""

    action: Literal["accept"]
    content: MCPFormContent


class MCPUrlElicitationAcceptedResponse(TypedDict):
    """Consent to a URL-mode interaction without passing user data to the server."""

    action: Literal["accept"]


class MCPElicitationDeclinedResponse(TypedDict):
    """An explicit user refusal of an elicitation request."""

    action: Literal["decline"]


class MCPElicitationCancelledResponse(TypedDict):
    """A user dismissal of an elicitation request without a decision."""

    action: Literal["cancel"]


MCPFormElicitationResponse: TypeAlias = (
    MCPFormElicitationAcceptedResponse
    | MCPElicitationDeclinedResponse
    | MCPElicitationCancelledResponse
)
"""A valid response to a form-mode elicitation interrupt."""

MCPUrlElicitationResponse: TypeAlias = (
    MCPUrlElicitationAcceptedResponse
    | MCPElicitationDeclinedResponse
    | MCPElicitationCancelledResponse
)
"""A valid response to a URL-mode elicitation interrupt."""

MCPElicitationResponse: TypeAlias = MCPFormElicitationResponse | MCPUrlElicitationResponse
"""A valid response to an MCP elicitation interrupt of either mode."""

MCPElicitationResume: TypeAlias = MCPElicitationResponse | dict[str, MCPElicitationResponse]
"""One elicitation response or a map of responses keyed by MCP input request ID."""


def interrupt_for_elicitation(
    params: _ElicitRequestParams,
    context: CallbackContext,
) -> _ElicitResult:
    """Pause a tool execution for a server-initiated MCP elicitation request.

    Args:
        params: The elicitation request received from the MCP server.
        context: Provenance for the server and tool issuing the request.

    Returns:
        The MCP result constructed from the value passed to `Command(resume=...)`.

    Raises:
        ToolException: If the server request or resumed user response is invalid.
    """
    mode = getattr(params, "mode", "form")
    if mode == "form":
        response_schema = getattr(params, "requested_schema", None)
        if not isinstance(response_schema, dict):
            msg = "MCP form elicitation must include an object response schema."
            raise ToolException(msg)
        payload: MCPElicitation = {
            "type": "mcp_elicitation",
            "mode": "form",
            "server": context.server_name,
            "message": params.message,
            "response_schema": response_schema,
        }
    elif mode == "url":
        url = getattr(params, "url", None)
        if url is None:
            msg = "MCP URL elicitation must include a URL."
            raise ToolException(msg)
        payload = {
            "type": "mcp_elicitation",
            "mode": "url",
            "server": context.server_name,
            "message": params.message,
            "url": str(url),
        }
    else:
        msg = f"MCP elicitation mode {mode!r} is not supported."
        raise ToolException(msg)

    resume_value = cast("MCPElicitationResponse", interrupt(payload))
    return _to_mcp_result(resume_value, mode=mode)


def _validate_form_content(content: Mapping[object, object]) -> MCPFormContent:
    """Validate content against the primitive MCP form-mode value subset."""
    validated: MCPFormContent = {}
    for key, value in content.items():
        if not isinstance(key, str):
            msg = "MCP form elicitation content keys must be strings."
            raise ToolException(msg)
        if (
            value is None
            or isinstance(value, str | int | float | bool)
            or (isinstance(value, list) and all(isinstance(item, str) for item in value))
        ):
            validated[key] = value
        else:
            msg = "MCP form elicitation content contains an unsupported value."
            raise ToolException(msg)
    return validated


def _to_mcp_result(resume_value: MCPElicitationResponse, *, mode: str) -> _ElicitResult:
    """Validate a cast LangGraph resume value and convert it to an MCP result."""
    if not isinstance(resume_value, Mapping):
        msg = "Resume an MCP elicitation with an action response object."
        raise ToolException(msg)

    action = resume_value.get("action")
    if action == "accept":
        if mode == "form":
            content = resume_value.get("content")
            if not isinstance(content, Mapping):
                msg = "Accepted MCP form elicitation requires object content."
                raise ToolException(msg)
            return _ElicitResult(action="accept", content=_validate_form_content(content))
        if "content" in resume_value:
            msg = "Accepted MCP URL elicitation must not include content."
            raise ToolException(msg)
        return _ElicitResult(action="accept")
    if action == "decline":
        if "content" in resume_value:
            msg = "Declined MCP elicitation must not include content."
            raise ToolException(msg)
        return _ElicitResult(action="decline")
    if action == "cancel":
        if "content" in resume_value:
            msg = "Cancelled MCP elicitation must not include content."
            raise ToolException(msg)
        return _ElicitResult(action="cancel")

    msg = "MCP elicitation responses require accept, decline, or cancel."
    raise ToolException(msg)

"""Convert MCP tools and elicitation requests into LangChain-native values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, cast

from langchain_core.tools import ToolException
from langgraph.types import interrupt
from typing_extensions import TypedDict


class MCPFormElicitation(TypedDict):
    """A structured form an MCP server asks the application to present."""

    type: Literal["mcp_elicitation"]
    mode: Literal["form"]
    message: str
    response_schema: dict[str, Any]


class MCPUrlElicitation(TypedDict):
    """A URL an MCP server asks the application to present for consent."""

    type: Literal["mcp_elicitation"]
    mode: Literal["url"]
    message: str
    url: str


MCPElicitation: TypeAlias = MCPFormElicitation | MCPUrlElicitation
MCPFormValue: TypeAlias = str | int | float | bool | list[str] | None
MCPFormContent: TypeAlias = dict[str, MCPFormValue]


class MCPFormElicitationAcceptedResponse(TypedDict):
    """An accepted form response with values matching the requested schema."""

    action: Literal["accept"]
    content: MCPFormContent


class MCPUrlElicitationAcceptedResponse(TypedDict):
    """Consent to a URL-mode interaction without passing content to the server."""

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


class MCPElicitationResponses(TypedDict):
    """Responses to multiple MCP elicitation requests, keyed by request ID."""

    responses: dict[str, MCPElicitationResponse]


MCPElicitationResume: TypeAlias = MCPElicitationResponse | MCPElicitationResponses
"""One elicitation response or a named collection of elicitation responses."""


def _normalize_input_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a provider-compatible copy of a remote MCP input schema."""
    normalized = dict(schema or {})
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    return normalized


def _tool_input_schema(tool: Any) -> Mapping[str, Any] | None:
    """Read the schema across FastMCP's supported tool model versions."""
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return schema if isinstance(schema, Mapping) else None


def _tool_metadata(tool: Any) -> dict[str, Any] | None:
    """Keep server-controlled tool fields in a non-colliding metadata namespace."""
    model_dump = getattr(tool, "model_dump", None)
    if not callable(model_dump):
        return None

    dumped = model_dump(by_alias=True, exclude_none=True)
    if not isinstance(dumped, dict):
        return None

    metadata = {
        key: dumped[key]
        for key in ("annotations", "title", "outputSchema", "icons", "execution", "_meta")
        if key in dumped
    }
    return {"mcp": metadata} if metadata else None


def _tool_result_content(result: Any) -> Any:
    """Extract model-visible content from a FastMCP tool result."""
    content = getattr(result, "content", None)
    if content is None:
        return str(getattr(result, "data", result))
    if isinstance(content, str):
        return content

    converted: list[Any] = []
    for block in content:
        model_dump = getattr(block, "model_dump", None)
        converted.append(model_dump(by_alias=True) if callable(model_dump) else block)
    return converted


def _tool_error_message(result: Any) -> str | None:
    """Return an MCP-declared tool error without exposing arbitrary result metadata."""
    model_dump = getattr(result, "model_dump", None)
    dumped = model_dump(by_alias=True) if callable(model_dump) else {}
    is_error = dumped.get("isError", False) if isinstance(dumped, dict) else False
    if not is_error:
        return None

    content = getattr(result, "content", ())
    messages: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            messages.append(text)
    return "\n".join(messages) or "The MCP server reported that the tool call failed."


def _tool_result_artifact(result: Any) -> dict[str, Any] | None:
    """Extract non-model-facing structured data from a FastMCP result."""
    model_dump = getattr(result, "model_dump", None)
    dumped = model_dump(by_alias=True) if callable(model_dump) else {}
    structured_content = dumped.get("structuredContent") if isinstance(dumped, dict) else None
    if structured_content is None:
        data = getattr(result, "data", None)
        if isinstance(data, (dict, list)):
            structured_content = data
    if structured_content is None:
        return None
    return {"mcp": {"structured_content": structured_content}}


def interrupt_for_elicitation(params: Any) -> dict[str, Any]:
    """Pause tool execution for a server-initiated MCP elicitation request.

    Args:
        params: FastMCP's form- or URL-mode elicitation request parameters.

    Returns:
        An elicitation result mapping accepted by FastMCP.

    Raises:
        ToolException: If the request or resumed response is invalid.
    """
    mode = getattr(params, "mode", "form")
    message = getattr(params, "message", None)
    if not isinstance(message, str):
        msg = "MCP elicitation must include a text message."
        raise ToolException(msg)

    if mode == "form":
        response_schema = getattr(params, "requested_schema", None)
        if response_schema is None:
            response_schema = getattr(params, "requestedSchema", None)
        if not isinstance(response_schema, dict):
            msg = "MCP form elicitation must include an object response schema."
            raise ToolException(msg)
        payload: MCPElicitation = {
            "type": "mcp_elicitation",
            "mode": "form",
            "message": message,
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
            "message": message,
            "url": str(url),
        }
    else:
        msg = f"MCP elicitation mode {mode!r} is not supported."
        raise ToolException(msg)

    resume_value = cast("MCPElicitationResume", interrupt(payload))
    return _to_fastmcp_result(resume_value, mode=mode)


def _to_fastmcp_result(resume_value: object, *, mode: str) -> dict[str, Any]:
    """Validate a LangGraph resume value and convert it to a FastMCP result."""
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
            return {"action": "accept", "content": _validate_form_content(content)}
        if "content" in resume_value:
            msg = "Accepted MCP URL elicitation must not include content."
            raise ToolException(msg)
        return {"action": "accept"}

    if action in {"decline", "cancel"}:
        if "content" in resume_value:
            msg = f"{action.capitalize()}d MCP elicitation must not include content."
            raise ToolException(msg)
        return {"action": action}

    msg = "MCP elicitation responses require accept, decline, or cancel."
    raise ToolException(msg)


def _validate_form_content(content: Mapping[object, object]) -> MCPFormContent:
    """Validate content against the MCP form-mode primitive value subset."""
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


__all__ = [
    "MCPElicitation",
    "MCPElicitationResponse",
    "MCPElicitationResponses",
    "MCPElicitationResume",
    "MCPFormElicitation",
    "MCPFormElicitationResponse",
    "MCPUrlElicitation",
    "MCPUrlElicitationResponse",
]

"""Convert MCP tool models and results into LangChain-native values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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

"""Extract natural-language text from LangChain tools."""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING, Any

from langchain_core.utils.pydantic import get_fields, is_basemodel_subclass

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.tools.base import BaseTool

# Credential-related terms for cross-modal correlation.
CREDENTIAL_NL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bapi[_ -]?key\b", re.IGNORECASE),
    re.compile(r"\baccess[_ -]?token\b", re.IGNORECASE),
    re.compile(r"\bauth(?:entication)?[_ -]?(?:token|header|key)\b", re.IGNORECASE),
    re.compile(r"\bcredential(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bpassword\b", re.IGNORECASE),
    re.compile(r"\bsecret(?:\s+key)?\b", re.IGNORECASE),
    re.compile(r"\bbearer[_ -]?token\b", re.IGNORECASE),
    re.compile(r"\bpass\s+the\s+(?:key|token|secret|credential)\b", re.IGNORECASE),
    re.compile(r"\binclude\s+(?:the\s+)?(?:key|token|secret)\b", re.IGNORECASE),
    re.compile(r"\breturn\s+(?:the\s+)?(?:key|token|secret|credential)\b", re.IGNORECASE),
    re.compile(r"\bfor\s+debug(?:ging)?\b", re.IGNORECASE),
)


def nl_mentions_credentials(text: str) -> bool:
    """Return whether natural-language text references credentials."""
    return any(pattern.search(text) for pattern in CREDENTIAL_NL_PATTERNS)


def _schema_field_descriptions(args_schema: Any) -> list[str]:
    """Extract description strings from a Pydantic args schema."""
    if args_schema is None:
        return []
    if isinstance(args_schema, dict):
        props = args_schema.get("properties", {})
        return [
            str(prop.get("description", ""))
            for prop in props.values()
            if isinstance(prop, dict) and prop.get("description")
        ]
    if is_basemodel_subclass(args_schema):
        descriptions: list[str] = []
        for field_info in get_fields(args_schema).values():
            description = getattr(field_info, "description", None)
            if description:
                descriptions.append(str(description))
        return descriptions
    return []


def extract_nl_from_tool(tool: BaseTool) -> str:
    """Collect natural-language text exposed to the LLM for a tool."""
    parts: list[str] = []
    if tool.description:
        parts.append(tool.description)
    parts.extend(_schema_field_descriptions(tool.args_schema))

    func = _get_tool_callable(tool)
    if func is not None:
        doc = inspect.getdoc(func)
        if doc:
            parts.append(doc)

    return "\n".join(parts)


def _get_tool_callable(tool: BaseTool) -> Callable[..., Any] | None:
    """Return the underlying Python callable for a tool, if available."""
    func = getattr(tool, "func", None)
    if callable(func):
        return func
    coroutine = getattr(tool, "coroutine", None)
    if callable(coroutine):
        return coroutine
    run_method = getattr(tool, "_run", None)
    if callable(run_method) and run_method.__func__ is not None:  # type: ignore[attr-defined]
        return run_method  # type: ignore[return-value]
    return None

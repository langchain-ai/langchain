"""Schema dataclass for LangChain tool definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import TypeAdapter


@dataclass
class ToolSchema:
    """Unified schema representation for a LangChain tool.

    This is the single source of truth for a tool's schema, validation, and
    token-estimation data. `BaseTool.tool_schema` is the one cached property;
    `tool_call_schema`, `args`, and `_approximate_schema_chars` are plain
    properties that delegate here.

    Attributes:
        name: The tool name.
        description: The tool description sent to the LLM.
        validator: A `TypeAdapter` for validating and coercing tool call inputs.
        json_schema: Pre-computed JSON schema dict describing the tool's
            parameters, suitable for passing directly to an LLM's tool/function
            calling API.
        pydantic_schema: The Pydantic model class or dict that backs
            `json_schema`. Preserved for backward compatibility with callers of
            `tool_call_schema` that check `issubclass(schema, BaseModel)`.
        args: Pre-computed properties dict (the `"properties"` field of
            `json_schema`), used by `BaseTool.args`.
        approximate_chars: Pre-computed char count of the neutral tool payload
            (name + description + schema), used for token estimation.
    """

    name: str
    description: str
    validator: TypeAdapter
    json_schema: dict[str, Any]
    pydantic_schema: Any
    args: dict[str, Any]
    approximate_chars: int

    def validate_python(self, data: Any) -> Any:
        """Validate and coerce tool call input data.

        Args:
            data: Raw input data to validate.

        Returns:
            Validated data, coerced to the expected types.
        """
        return self.validator.validate_python(data)

"""Types for setting agent response formats."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, is_dataclass
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, TypeAdapter
from typing_extensions import Self, is_typeddict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from langchain_core.messages import AIMessage

# Supported schema types: Pydantic models, dataclasses, TypedDict, JSON schema dicts
SchemaT = TypeVar("SchemaT")

SchemaKind = Literal["pydantic", "dataclass", "typeddict", "json_schema"]


class StructuredOutputError(Exception):
    """Base class for structured output errors."""

    ai_message: AIMessage


class MultipleStructuredOutputsError(StructuredOutputError):
    """Raised when model returns multiple structured output tool calls when only one is expected."""

    def __init__(self, tool_names: list[str], ai_message: AIMessage) -> None:
        """Initialize `MultipleStructuredOutputsError`.

        Args:
            tool_names: The names of the tools called for structured output.
            ai_message: The AI message that contained the invalid multiple tool calls.
        """
        self.tool_names = tool_names
        self.ai_message = ai_message

        super().__init__(
            "Model incorrectly returned multiple structured responses "
            f"({', '.join(tool_names)}) when only one is expected."
        )


class StructuredOutputValidationError(StructuredOutputError):
    """Raised when structured output tool call arguments fail to parse according to the schema."""

    def __init__(self, tool_name: str, source: Exception, ai_message: AIMessage) -> None:
        """Initialize `StructuredOutputValidationError`.

        Args:
            tool_name: The name of the tool that failed.
            source: The exception that occurred.
            ai_message: The AI message that contained the invalid structured output.
        """
        self.tool_name = tool_name
        self.source = source
        self.ai_message = ai_message
        super().__init__(f"Failed to parse structured output for tool '{tool_name}': {source}.")


def _parse_with_schema(
    schema: type[SchemaT] | dict, schema_kind: SchemaKind, data: dict[str, Any]
) -> Any:
    """Parse data using for any supported schema type.

    Args:
        schema: The schema type (Pydantic model, `dataclass`, or `TypedDict`)
        schema_kind: One of `"pydantic"`, `"dataclass"`, `"typeddict"`, or
            `"json_schema"`
        data: The data to parse

    Returns:
        The parsed instance according to the schema type

    Raises:
        ValueError: If parsing fails
    """
    if schema_kind == "json_schema":
        return data
    try:
        adapter: TypeAdapter[SchemaT] = TypeAdapter(schema)
        return adapter.validate_python(data)
    except Exception as e:
        schema_name = getattr(schema, "__name__", str(schema))
        msg = f"Failed to parse data to {schema_name}: {e}"
        raise ValueError(msg) from e


@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema."""

    schema: type[SchemaT]
    """The schema for the response, can be a Pydantic model, `dataclass`, `TypedDict`,
    or JSON schema dict."""

    name: str
    """Name of the schema, used for tool calling.

    If not provided, the name will be the model name or `"response_format"` if it's a
    JSON schema.
    """

    description: str
    """Custom description of the schema.

    If not provided, provided will use the model's docstring.
    """

    schema_kind: SchemaKind
    """The kind of schema."""

    json_schema: dict[str, Any]
    """JSON schema associated with the schema."""

    strict: bool | None = None
    """Whether to enforce strict validation of the schema."""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> None:
        """Initialize SchemaSpec with schema and optional parameters."""
        self.schema = schema

        if name:
            self.name = name
        elif isinstance(schema, dict):
            self.name = str(schema.get("title", f"response_format_{str(uuid.uuid4())[:4]}"))
        else:
            self.name = str(getattr(schema, "__name__", f"response_format_{str(uuid.uuid4())[:4]}"))

        self.description = description or (
            schema.get("description", "")
            if isinstance(schema, dict)
            else getattr(schema, "__doc__", None) or ""
        )

        self.strict = strict

        if isinstance(schema, dict):
            self.schema_kind = "json_schema"
            self.json_schema = schema
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            self.schema_kind = "pydantic"
            self.json_schema = schema.model_json_schema()
        elif is_dataclass(schema):
            self.schema_kind = "dataclass"
            self.json_schema = TypeAdapter(schema).json_schema()
        elif is_typeddict(schema):
            self.schema_kind = "typeddict"
            self.json_schema = TypeAdapter(schema).json_schema()
        else:
            msg = (
                f"Unsupported schema type: {type(schema)}. "
                f"Supported types: Pydantic models, dataclasses, TypedDicts, and JSON schema dicts."
            )
            raise ValueError(msg)


@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
    """Use a tool calling strategy for model responses."""

    schema: type[SchemaT]
    """Schema for the tool calls."""

    schema_specs: list[_SchemaSpec[SchemaT]]
    """Schema specs for the tool calls."""

    tool_message_content: str | None
    """The content of the tool message to be returned when the model calls
    an artificial structured output tool."""

    handle_errors: (
        bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]
    )
    """Error handling strategy for structured output via `ToolStrategy`.

    - `True`: Catch all errors with default error template
    - `str`: Catch all errors with this custom message
    - `type[Exception]`: Only catch this exception type with default message
    - `tuple[type[Exception], ...]`: Only catch these exception types with default
        message
    - `Callable[[Exception], str]`: Custom function that returns error message
    - `False`: No retry, let exceptions propagate
    """

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        tool_message_content: str | None = None,
        handle_errors: bool
        | str
        | type[Exception]
        | tuple[type[Exception], ...]
        | Callable[[Exception], str] = True,
    ) -> None:
        """Initialize `ToolStrategy`.

        Initialize `ToolStrategy` with schemas, tool message content, and error handling
        strategy.
        """
        self.schema = schema
        self.tool_message_content = tool_message_content
        self.handle_errors = handle_errors

        def _iter_variants(schema: Any) -> Iterable[Any]:
            """Yield leaf variants from Union and JSON Schema oneOf."""
            if get_origin(schema) in {UnionType, Union}:
                for arg in get_args(schema):
                    yield from _iter_variants(arg)
                return

            if isinstance(schema, dict) and "oneOf" in schema:
                for sub in schema.get("oneOf", []):
                    yield from _iter_variants(sub)
                return

            yield schema

        self.schema_specs = [_SchemaSpec(s) for s in _iter_variants(schema)]


@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
    """Use the model provider's native structured output method."""

    schema: type[SchemaT]
    """Schema for native mode."""

    schema_spec: _SchemaSpec[SchemaT]
    """Schema spec for native mode."""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        strict: bool | None = None,
    ) -> None:
        """Initialize ProviderStrategy with schema.

        Args:
            schema: Schema to enforce via the provider's native structured output.
            strict: Whether to request strict provider-side schema enforcement.
        """
        self.schema = schema
        self.schema_spec = _SchemaSpec(schema, strict=strict)

    def to_model_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs to bind to a model to force structured output."""
        # OpenAI:
        # - see https://platform.openai.com/docs/guides/structured-outputs
        json_schema: dict[str, Any] = {
            "name": self.schema_spec.name,
            "schema": self.schema_spec.json_schema,
        }
        if self.schema_spec.strict:
            json_schema["strict"] = True

        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        return {"response_format": response_format}


@dataclass
class OutputToolBinding(Generic[SchemaT]):
    """Information for tracking structured output tool metadata.

    This contains all necessary information to handle structured responses
    generated via tool calls, including the original schema, its type classification,
    and the corresponding tool implementation used by the tools strategy.
    """

    schema: type[SchemaT]
    """The original schema provided for structured output
    (Pydantic model, dataclass, TypedDict, or JSON schema dict)."""

    schema_kind: SchemaKind
    """Classification of the schema type for proper response construction."""

    tool: BaseTool
    """LangChain tool instance created from the schema for model binding."""

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create an `OutputToolBinding` instance from a `SchemaSpec`.

        Args:
            schema_spec: The `SchemaSpec` to convert

        Returns:
            An `OutputToolBinding` instance with the appropriate tool created
        """
        return cls(
            schema=schema_spec.schema,
            schema_kind=schema_spec.schema_kind,
            tool=StructuredTool(
                args_schema=schema_spec.json_schema,
                name=schema_spec.name,
                description=schema_spec.description,
            ),
        )

    def parse(self, tool_args: dict[str, Any]) -> SchemaT:
        """Parse tool arguments according to the schema.

        Args:
            tool_args: The arguments from the tool call

        Returns:
            The parsed response according to the schema type

        Raises:
            ValueError: If parsing fails
        """
        return _parse_with_schema(self.schema, self.schema_kind, tool_args)


@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    """Information for tracking native structured output metadata.

    This contains all necessary information to handle structured responses
    generated via native provider output, including the original schema,
    its type classification, and parsing logic for provider-enforced JSON.
    """

    schema: type[SchemaT]
    """The original schema provided for structured output
    (Pydantic model, `dataclass`, `TypedDict`, or JSON schema dict)."""

    schema_kind: SchemaKind
    """Classification of the schema type for proper response construction."""

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create a `ProviderStrategyBinding` instance from a `SchemaSpec`.

        Args:
            schema_spec: The `SchemaSpec` to convert

        Returns:
            A `ProviderStrategyBinding` instance for parsing native structured output
        """
        return cls(
            schema=schema_spec.schema,
            schema_kind=schema_spec.schema_kind,
        )

    def parse(self, response: AIMessage) -> SchemaT:
        """Parse `AIMessage` content according to the schema.

        Args:
            response: The `AIMessage` containing the structured output

        Returns:
            The parsed response according to the schema

        Raises:
            ValueError: If text extraction, JSON parsing or schema validation fails
        """
        # Extract text content from AIMessage and parse as JSON
        raw_text = self._extract_text_content_from_message(response)

        import json

        try:
            data = json.loads(raw_text)
        except Exception as e:
            schema_name = getattr(self.schema, "__name__", "response_format")
            msg = (
                f"Native structured output expected valid JSON for {schema_name}, "
                f"but parsing failed: {e}."
            )
            raise ValueError(msg) from e

        # Parse according to schema
        return _parse_with_schema(self.schema, self.schema_kind, data)

    def _extract_text_content_from_message(self, message: AIMessage) -> str:
        """Extract text content from an AIMessage.

        Args:
            message: The AI message to extract text from

        Returns:
            The extracted text content
        """
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text" and "text" in c:
                        parts.append(str(c["text"]))
                    elif "content" in c and isinstance(c["content"], str):
                        parts.append(c["content"])
                else:
                    parts.append(str(c))
            return "".join(parts)
        return str(content)


class AutoStrategy(Generic[SchemaT]):
    """Automatically select the best strategy for structured output."""

    schema: type[SchemaT]
    """Schema for automatic mode."""

    def __init__(
        self,
        schema: type[SchemaT],
    ) -> None:
        """Initialize AutoStrategy with schema."""
        self.schema = schema


ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]

"""Tool argument validation middleware for agents."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ValidationError
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.tools import BaseTool

_MIXED_BATCH_ERROR = (
    "Tool call batch rejected because another tool call in the same assistant message "
    "had invalid arguments. Regenerate the full set of tool calls from your previous "
    "response."
)

ValidatorKind = Literal["json_schema", "pydantic_v1", "pydantic_v2"]


@dataclass
class _ToolValidatorSpec:
    """Cached validation metadata for a tool."""

    tool: BaseTool
    kind: ValidatorKind
    schema: type[BaseModel | BaseModelV1] | dict[str, Any]
    field_names: frozenset[str] | None = None
    json_schema_validator: Any | None = None


@dataclass
class _ValidatedToolCall:
    """Validation result for a single tool call."""

    tool_call: dict[str, Any]
    error_message: str | None = None


class ToolArgValidationMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Validate LLM-generated tool arguments before tool execution."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        strip_empty_values: bool = True,
        json_schema_validator_class: type[Any] | None = None,
    ) -> None:
        """Initialize the tool argument validation middleware.

        Args:
            max_retries: Maximum number of validation retries after the initial model
                response.
            strip_empty_values: Whether to recursively remove empty dict keys with
                `None`, `{}`, or `[]` values before validation.
            json_schema_validator_class: Optional JSON Schema validator class for dict
                schemas. Defaults to `jsonschema.Draft7Validator` when needed.

        Raises:
            TypeError: If `json_schema_validator_class` is not a class.
            ValueError: If `max_retries` is negative.
        """
        super().__init__()
        if max_retries < 0:
            msg = "max_retries must be >= 0"
            raise ValueError(msg)
        if json_schema_validator_class is not None and not isinstance(
            json_schema_validator_class, type
        ):
            msg = "json_schema_validator_class must be a class or None"
            raise TypeError(msg)

        self.max_retries = max_retries
        self.strip_empty_values = strip_empty_values
        self.json_schema_validator_class = json_schema_validator_class
        self.tools = []
        self._validator_cache: dict[str, _ToolValidatorSpec] = {}

    def _get_json_schema_validator_class(self) -> type[Any]:
        """Resolve the JSON Schema validator class lazily."""
        if self.json_schema_validator_class is not None:
            return self.json_schema_validator_class

        try:
            jsonschema = importlib.import_module("jsonschema")
        except ImportError as exc:
            msg = (
                "ToolArgValidationMiddleware requires the jsonschema package to "
                "validate tools with dict args_schema. Please install it with "
                "`pip install jsonschema` or pass json_schema_validator_class."
            )
            raise ImportError(msg) from exc

        return cast("type[Any]", jsonschema.Draft7Validator)

    def _build_validator_spec(self, tool: BaseTool) -> _ToolValidatorSpec:
        """Build a validator spec for a tool."""
        schema = tool.tool_call_schema

        if isinstance(schema, dict):
            properties = schema.get("properties", {})
            validator_class = self._get_json_schema_validator_class()
            return _ToolValidatorSpec(
                tool=tool,
                kind="json_schema",
                schema=schema,
                field_names=(
                    frozenset(properties.keys()) if isinstance(properties, dict) else None
                ),
                json_schema_validator=validator_class(schema),
            )

        if issubclass(schema, BaseModelV1):
            return _ToolValidatorSpec(
                tool=tool,
                kind="pydantic_v1",
                schema=schema,
                field_names=frozenset(schema.__fields__),
            )

        if issubclass(schema, BaseModel):
            return _ToolValidatorSpec(
                tool=tool,
                kind="pydantic_v2",
                schema=schema,
                field_names=frozenset(schema.model_fields),
            )

        msg = f"Unsupported tool_call_schema for tool '{tool.name}': {schema!r}"
        raise TypeError(msg)

    def _resolve_request_validators(
        self, tools: Sequence[BaseTool | dict[str, Any]]
    ) -> dict[str, _ToolValidatorSpec]:
        """Resolve validators for the current request tools."""
        resolved: dict[str, _ToolValidatorSpec] = {}

        for tool in tools:
            if isinstance(tool, dict):
                continue

            cached_spec = self._validator_cache.get(tool.name)
            if cached_spec is None or cached_spec.tool is not tool:
                cached_spec = self._build_validator_spec(tool)
                self._validator_cache[tool.name] = cached_spec

            resolved[tool.name] = cached_spec

        return resolved

    @staticmethod
    def _format_path(path: Sequence[Any]) -> str:
        """Format a nested error path."""
        if not path:
            return "<root>"
        return ".".join(str(part) for part in path)

    @staticmethod
    def _build_error_message(
        tool_name: str,
        args: dict[str, Any] | Any,
        details: list[str],
    ) -> str:
        """Build a validation error message for a tool call."""
        rendered_args = repr(args)
        joined_details = "\n".join(f"- {detail}" for detail in details) or "- Invalid input"
        return (
            f"Invalid arguments for tool '{tool_name}'. Regenerate the full set of tool "
            f"calls from your previous response.\n"
            f"Received args: {rendered_args}\n"
            f"Validation errors:\n{joined_details}"
        )

    @staticmethod
    def _filter_args_for_message(
        args: dict[str, Any], field_names: frozenset[str] | None
    ) -> dict[str, Any]:
        """Filter rendered args to fields declared in the tool schema."""
        if field_names is None:
            return args
        return {key: value for key, value in args.items() if key in field_names}

    def _strip_empty_value(self, value: Any) -> Any:
        """Recursively remove empty dict values while preserving list shape."""
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, child in value.items():
                cleaned_child = self._strip_empty_value(child)
                if cleaned_child is None:
                    continue
                if isinstance(cleaned_child, dict) and not cleaned_child:
                    continue
                if isinstance(cleaned_child, list) and not cleaned_child:
                    continue
                cleaned[key] = cleaned_child
            return cleaned

        if isinstance(value, list):
            return [self._strip_empty_value(item) for item in value]

        return value

    def _sanitize_args(self, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Return sanitized tool arguments for validation."""
        cleaned_args = cast("dict[str, Any]", self._strip_empty_value(tool_args))
        return cleaned_args if self.strip_empty_values else tool_args.copy()

    def _format_pydantic_error(
        self,
        validator_spec: _ToolValidatorSpec,
        args: dict[str, Any],
        exc: ValidationError | ValidationErrorV1,
    ) -> str:
        """Format a Pydantic validation error."""
        details = [
            f"{self._format_path(cast('Sequence[Any]', err.get('loc', ())))}: "
            f"{err.get('msg', 'Invalid value')}"
            for err in exc.errors()
        ]
        display_args = self._filter_args_for_message(args, validator_spec.field_names)
        return self._build_error_message(validator_spec.tool.name, display_args, details)

    def _format_json_schema_error(
        self,
        validator_spec: _ToolValidatorSpec,
        args: dict[str, Any],
        errors: list[Any],
    ) -> str:
        """Format JSON Schema validation errors."""
        details = [f"{self._format_path(error.path)}: {error.message}" for error in errors]
        display_args = self._filter_args_for_message(args, validator_spec.field_names)
        return self._build_error_message(validator_spec.tool.name, display_args, details)

    def _validate_known_tool_call(
        self,
        tool_call: dict[str, Any],
        validator_spec: _ToolValidatorSpec,
    ) -> _ValidatedToolCall:
        """Validate a tool call against a resolved tool schema."""
        raw_args = tool_call.get("args")
        if not isinstance(raw_args, dict):
            error_message = self._build_error_message(
                validator_spec.tool.name,
                raw_args,
                [f"<root>: expected an object but received {type(raw_args).__name__}"],
            )
            return _ValidatedToolCall(tool_call=tool_call, error_message=error_message)

        sanitized_args = self._sanitize_args(raw_args)
        tool_call["args"] = sanitized_args

        try:
            if validator_spec.kind == "pydantic_v2":
                cast("type[BaseModel]", validator_spec.schema).model_validate(sanitized_args)
            elif validator_spec.kind == "pydantic_v1":
                cast("type[BaseModelV1]", validator_spec.schema).parse_obj(sanitized_args)
            else:
                json_schema_validator = validator_spec.json_schema_validator
                errors = sorted(
                    json_schema_validator.iter_errors(sanitized_args),
                    key=lambda error: list(error.path),
                )
                if errors:
                    error_message = self._format_json_schema_error(
                        validator_spec,
                        sanitized_args,
                        errors,
                    )
                    return _ValidatedToolCall(tool_call=tool_call, error_message=error_message)
        except (ValidationError, ValidationErrorV1) as exc:
            error_message = self._format_pydantic_error(validator_spec, sanitized_args, exc)
            return _ValidatedToolCall(tool_call=tool_call, error_message=error_message)

        return _ValidatedToolCall(tool_call=tool_call)

    def _build_retry_messages(
        self,
        validated_tool_calls: list[_ValidatedToolCall],
    ) -> list[ToolMessage]:
        """Build synthetic tool messages for a rejected tool-call batch."""
        has_invalid_tool_call = any(
            result.error_message is not None for result in validated_tool_calls
        )
        if not has_invalid_tool_call:
            return []

        tool_messages: list[ToolMessage] = []
        for result in validated_tool_calls:
            tool_call = result.tool_call
            content = result.error_message or _MIXED_BATCH_ERROR
            tool_messages.append(
                ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"] or "",
                    status="error",
                )
            )

        return tool_messages

    def _validate_model_response(
        self,
        model_response: ModelResponse[ResponseT],
        request: ModelRequest[ContextT],
    ) -> tuple[ModelResponse[ResponseT], list[ToolMessage], bool]:
        """Validate a model response and decide whether to retry."""
        if len(model_response.result) != 1:
            return model_response, [], True

        message = model_response.result[0]
        if not isinstance(message, AIMessage) or not message.tool_calls:
            return model_response, [], True

        validators_by_name = self._resolve_request_validators(request.tools)
        if not validators_by_name:
            return model_response, [], True

        ai_message = message.model_copy(deep=True)
        validated_tool_calls: list[_ValidatedToolCall] = []

        for tool_call in ai_message.tool_calls:
            validator_spec = validators_by_name.get(tool_call["name"])
            if validator_spec is None:
                validated_tool_calls.append(_ValidatedToolCall(tool_call=tool_call))
                continue

            validated_tool_calls.append(self._validate_known_tool_call(tool_call, validator_spec))

        tool_messages = self._build_retry_messages(validated_tool_calls)
        if tool_messages:
            return (
                ModelResponse(
                    result=[ai_message],
                    structured_response=model_response.structured_response,
                ),
                tool_messages,
                False,
            )

        return (
            ModelResponse(
                result=[ai_message],
                structured_response=model_response.structured_response,
            ),
            [],
            True,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Validate tool arguments around synchronous model execution."""
        current_request = request

        for attempt in range(self.max_retries + 1):
            model_response = handler(current_request)
            validated_response, tool_messages, is_valid = self._validate_model_response(
                model_response, current_request
            )
            if is_valid:
                return validated_response
            if attempt == self.max_retries:
                return model_response

            ai_message = cast("AIMessage", validated_response.result[0].model_copy(deep=True))
            current_request = current_request.override(
                messages=[*current_request.messages, ai_message, *tool_messages]
            )

        msg = "Unexpected: validation retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Validate tool arguments around asynchronous model execution."""
        current_request = request

        for attempt in range(self.max_retries + 1):
            model_response = await handler(current_request)
            validated_response, tool_messages, is_valid = self._validate_model_response(
                model_response, current_request
            )
            if is_valid:
                return validated_response
            if attempt == self.max_retries:
                return model_response

            ai_message = cast("AIMessage", validated_response.result[0].model_copy(deep=True))
            current_request = current_request.override(
                messages=[*current_request.messages, ai_message, *tool_messages]
            )

        msg = "Unexpected: validation retry loop completed without returning"
        raise RuntimeError(msg)

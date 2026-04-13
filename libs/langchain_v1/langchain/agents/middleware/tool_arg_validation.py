r"""Tool argument validation middleware for agents.

Validates LLM-generated tool arguments against each tool's schema **before**
the tool node (and any human-in-the-loop middleware) fires.

Uses ``wrap_model_call`` / ``awrap_model_call`` to wrap the model invocation.
When the model generates tool calls with invalid arguments, the middleware
appends error ``ToolMessage``\s to the conversation and re-invokes the model
so it can regenerate corrected arguments.  The retry loop runs entirely inside
the model node so only the final valid ``AIMessage`` ever enters the graph
state.

Both MCP tools (``args_schema`` is a ``dict``) and Pydantic-based tools
(``args_schema`` is a ``BaseModel`` subclass) are validated.  MCP tools
require the ``jsonschema`` package; Pydantic tools use ``model_validate``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #


@dataclass
class _ValidationError:
    """Uniform representation of a single validation error."""

    path: str
    message: str


def _strip_empty_values(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove keys whose value is ``None``, ``{}``, or ``[]``.

    LLMs (especially Gemini) emit explicit ``null`` or empty containers
    for optional fields instead of omitting them.  Stripping these
    prevents unnecessary validation retries — if the field is optional
    it simply becomes absent; if required the error changes to a clear
    ``'<field>' is a required property`` message.

    Args:
        d: Dictionary to strip empty values from.

    Returns:
        New dictionary with empty values removed.
    """
    result: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or v in ({}, []):
            continue
        if isinstance(v, dict):
            result[k] = _strip_empty_values(v)
        elif isinstance(v, list):
            result[k] = [
                _strip_empty_values(item) if isinstance(item, dict) else item for item in v
            ]
        else:
            result[k] = v
    return result


def _validate_with_pydantic(
    model_class: type[BaseModel], args: dict[str, Any]
) -> list[_ValidationError]:
    """Validate args against a Pydantic model.

    Args:
        model_class: Pydantic model class to validate against.
        args: Tool call arguments to validate.

    Returns:
        List of validation errors, empty if valid.
    """
    try:
        model_class.model_validate(args)
    except PydanticValidationError as exc:
        return [
            _ValidationError(
                path=" → ".join(str(p) for p in e["loc"]) or "(root)",
                message=e["msg"],
            )
            for e in exc.errors()
        ]
    else:
        return []


def _validate_with_json_schema(
    schema: dict[str, Any],
    args: dict[str, Any],
    *,
    validator_class: type[Any] | None = None,
) -> list[_ValidationError]:
    """Validate args against a JSON Schema using ``jsonschema``.

    Args:
        schema: JSON Schema dictionary.
        args: Tool call arguments to validate.
        validator_class: JSON Schema validator class to use.  Defaults to
            ``Draft7Validator`` if not provided.

    Returns:
        List of validation errors, empty if valid.

    Raises:
        ImportError: If ``jsonschema`` is not installed.
    """
    try:
        from jsonschema import Draft7Validator  # noqa: PLC0415
    except ImportError:
        msg = (
            "The 'jsonschema' package is required for validating MCP tools "
            "with dict-based schemas. Install it with: pip install jsonschema"
        )
        raise ImportError(msg)  # noqa: B904

    cls = validator_class or Draft7Validator
    validator = cls(schema)
    errors = list(validator.iter_errors(args))
    if not errors:
        return []
    return [
        _ValidationError(
            path=" → ".join(str(p) for p in err.absolute_path) or "(root)",
            message=err.message,
        )
        for err in errors
    ]


def _format_validation_errors(tool_name: str, errors: list[_ValidationError]) -> str:
    """Build a concise, LLM-friendly description of validation errors.

    Args:
        tool_name: Name of the tool that failed validation.
        errors: List of validation errors.

    Returns:
        Formatted error message string.
    """
    parts = [
        f"Tool '{tool_name}' argument validation failed. Fix the following errors and retry:",
        *[f"  • [{err.path}] {err.message}" for err in errors],
    ]
    parts.append(
        "\nHint: if a field is optional and not needed, omit it entirely "
        "from the arguments rather than setting it to null or an empty value."
    )
    return "\n".join(parts)


def _resolve_tool_schemas(
    tools: list[BaseTool | dict[str, Any]],
) -> dict[str, type[BaseModel] | dict[str, Any]]:
    """Build a mapping of tool name to schema for validatable tools.

    Tools with a Pydantic ``BaseModel`` as ``args_schema`` are stored as the
    model class.  Tools with a ``dict`` ``args_schema`` (MCP tools) are stored
    as the raw JSON Schema dict.  Raw dict tool specs and tools with no schema
    are skipped.

    Args:
        tools: List of tools from the model request.

    Returns:
        Mapping of tool name to either a Pydantic model class or JSON Schema dict.
    """
    schemas: dict[str, type[BaseModel] | dict[str, Any]] = {}
    for tool in tools:
        if isinstance(tool, dict):
            continue  # raw dict tool specs — no schema to validate against

        args_schema = tool.args_schema
        if args_schema is None:
            continue

        if isinstance(args_schema, dict) or (
            isinstance(args_schema, type) and issubclass(args_schema, BaseModel)
        ):
            schemas[tool.name] = args_schema

    return schemas


# ------------------------------------------------------------------ #
# Middleware class
# ------------------------------------------------------------------ #


class ToolArgValidationMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    r"""Validate tool call arguments before execution and retry on failure.

    Uses ``wrap_model_call`` / ``awrap_model_call`` to intercept the model's
    output inside the model node.  When the model generates tool calls with
    invalid arguments, the middleware appends error ``ToolMessage``\s to the
    conversation and re-invokes the model so it can regenerate corrected
    arguments.

    Only the final valid response is returned to the graph, ensuring:

    *   HITL middleware (``aafter_model``) only ever sees valid tool-call
        arguments.
    *   The tools node runs exactly once per valid tool call.
    *   No ``Command(goto)`` redirects that could cause duplicate execution.

    Both MCP tools (``args_schema`` is a ``dict``) and Pydantic-based tools
    (``args_schema`` is a ``BaseModel`` subclass) are validated.  MCP tools
    require the ``jsonschema`` package (soft dependency); Pydantic tools are
    validated via ``model_validate``.

    Before validation, keys with empty values (``None``, ``{}``, ``[]``) are
    optionally stripped from tool-call arguments.  LLMs (especially Gemini)
    routinely emit explicit ``null`` or empty containers for optional fields;
    stripping prevents unnecessary validation retries.

    Examples:
        !!! example "Basic usage with default settings"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ToolArgValidationMiddleware

            agent = create_agent(
                model,
                tools=[search_tool, mcp_tool],
                middleware=[ToolArgValidationMiddleware()],
            )
            ```

        !!! example "Custom retry count and no empty-value stripping"

            ```python
            validation = ToolArgValidationMiddleware(
                max_retries=3,
                strip_empty_values=False,
            )
            agent = create_agent(model, tools=tools, middleware=[validation])
            ```

        !!! example "Combined with HITL — validation runs first"

            ```python
            from langchain.agents.middleware import (
                HumanInTheLoopMiddleware,
                ToolArgValidationMiddleware,
            )

            agent = create_agent(
                model,
                tools=tools,
                middleware=[
                    ToolArgValidationMiddleware(),  # validates args
                    HumanInTheLoopMiddleware(...),  # then asks human
                ],
            )
            ```

    !!! warning
        MCP tools with ``dict``-based schemas require the ``jsonschema``
        package.  Install it with ``pip install jsonschema``.
    """

    def __init__(
        self,
        *,
        max_retries: int = 2,
        strip_empty_values: bool = True,
        json_schema_validator_class: type[Any] | None = None,
    ) -> None:
        """Initialize `ToolArgValidationMiddleware`.

        Args:
            max_retries: Maximum number of validation-retry cycles per model
                invocation. Must be ``>= 1``.
            strip_empty_values: Whether to recursively strip keys with ``None``,
                ``{}``, or ``[]`` values from tool-call arguments before validation.
            json_schema_validator_class: Optional JSON Schema validator class for
                dict-based schemas (MCP tools).  Must conform to the ``jsonschema``
                ``Validator`` protocol (e.g. ``Draft7Validator``,
                ``Draft202012Validator``).  Defaults to ``Draft7Validator``.

        Raises:
            ValueError: If ``max_retries < 1``.
        """
        super().__init__()

        if max_retries < 1:
            msg = f"max_retries must be >= 1, got {max_retries}"
            raise ValueError(msg)

        self.max_retries = max_retries
        self.strip_empty_values = strip_empty_values
        self.json_schema_validator_class = json_schema_validator_class
        self.tools: list[BaseTool] = []  # no additional tools
        self._schemas: dict[str, type[BaseModel] | dict[str, Any]] = {}
        self._schemas_resolved = False

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Call the model, validate tool args, and retry on failure.

        Args:
            request: Model request with model, messages, state, and runtime.
            handler: Callable to execute the model (can be called multiple times).

        Returns:
            ``ModelResponse`` or ``AIMessage`` with valid tool calls.
        """
        self._ensure_schemas(request.tools)
        response = handler(request)

        for attempt in range(1, self.max_retries + 1):
            ai_msg = _extract_ai_message(response)
            if ai_msg is None or not ai_msg.tool_calls:
                return response

            error_messages = self._validate_tool_calls(ai_msg)
            if not error_messages:
                return response

            logger.warning(
                "Validation failed (attempt %d/%d), retrying model call",
                attempt,
                self.max_retries,
            )

            retry_msgs: list[AnyMessage] = list(request.messages)
            retry_msgs.append(ai_msg)
            retry_msgs.extend(error_messages)
            response = handler(request.override(messages=retry_msgs))

        logger.warning(
            "Validation retries exhausted (%d), passing through",
            self.max_retries,
        )
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Async version of ``wrap_model_call``.

        Args:
            request: Model request with model, messages, state, and runtime.
            handler: Async callable to execute the model.

        Returns:
            ``ModelResponse`` or ``AIMessage`` with valid tool calls.
        """
        self._ensure_schemas(request.tools)
        response = await handler(request)

        for attempt in range(1, self.max_retries + 1):
            ai_msg = _extract_ai_message(response)
            if ai_msg is None or not ai_msg.tool_calls:
                return response

            error_messages = self._validate_tool_calls(ai_msg)
            if not error_messages:
                return response

            logger.warning(
                "Validation failed (attempt %d/%d), retrying model call",
                attempt,
                self.max_retries,
            )

            retry_msgs: list[AnyMessage] = list(request.messages)
            retry_msgs.append(ai_msg)
            retry_msgs.extend(error_messages)
            response = await handler(request.override(messages=retry_msgs))

        logger.warning(
            "Validation retries exhausted (%d), passing through",
            self.max_retries,
        )
        return response

    def _ensure_schemas(self, tools: list[BaseTool | dict[str, Any]]) -> None:
        """Populate the schema cache from request tools on first invocation.

        Args:
            tools: List of tools from the model request.
        """
        if self._schemas_resolved:
            return
        self._schemas = _resolve_tool_schemas(tools)
        self._schemas_resolved = True

    def _validate_tool_calls(self, ai_msg: AIMessage) -> list[ToolMessage]:
        r"""Validate all tool calls and return error ``ToolMessage``\s for failures.

        When any tool call is invalid, **all** tool calls receive a
        ``ToolMessage``: invalid ones get the error description; valid ones
        get a "not executed" notice.  This ensures every ``tool_call`` has a
        matching response (required by some providers, e.g. Gemini).

        Args:
            ai_msg: The ``AIMessage`` containing tool calls.

        Returns:
            Empty list if all tool calls are valid, otherwise a list of
            ``ToolMessage``\s (one per tool call).
        """
        error_messages: list[ToolMessage] = []
        valid_call_ids: list[str] = []

        for tc in ai_msg.tool_calls:
            tool_name: str = tc.get("name") or ""
            tool_args: dict[str, Any] = tc.get("args") or {}
            tool_call_id: str = tc.get("id") or ""

            # Skip tools we have no schema for
            if tool_name not in self._schemas:
                valid_call_ids.append(tool_call_id)
                continue

            # Optionally strip empty values before validation
            if self.strip_empty_values:
                tool_args = _strip_empty_values(tool_args)
                tc["args"] = tool_args

            schema = self._schemas[tool_name]
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                errors = _validate_with_pydantic(schema, tool_args)
            else:
                errors = _validate_with_json_schema(
                    schema, tool_args, validator_class=self.json_schema_validator_class
                )

            if not errors:
                valid_call_ids.append(tool_call_id)
                continue

            error_msg = _format_validation_errors(tool_name, errors)
            logger.warning(
                "Validation failed for tool '%s' (call_id=%s): %s",
                tool_name,
                tool_call_id,
                error_msg,
            )
            error_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id))

        if not error_messages:
            return []

        # Add "not executed" messages for valid tool calls in the same batch
        error_messages.extend(
            ToolMessage(
                content=(
                    "Tool call was not executed because other tool calls "
                    "in this batch failed argument validation. "
                    "Please retry all tool calls with corrected arguments."
                ),
                tool_call_id=call_id,
            )
            for call_id in valid_call_ids
        )

        return error_messages


def _extract_ai_message(response: ModelResponse[Any]) -> AIMessage | None:
    """Extract the ``AIMessage`` from a ``ModelResponse``, if present.

    Args:
        response: Model response to inspect.

    Returns:
        The ``AIMessage`` if found, otherwise ``None``.
    """
    for msg in response.result:
        if isinstance(msg, AIMessage):
            return msg
    return None

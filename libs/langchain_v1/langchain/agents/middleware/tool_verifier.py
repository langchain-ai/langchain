"""Tool verifier middleware for agents."""

from __future__ import annotations

from datetime import datetime, timezone
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    TracePolicy,
    omit_payload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest


class ToolVerifierMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Middleware that verifies tool calls before execution via external verifier.

    This middleware is opt-in. When configured, it calls a verifier function
    before each tool execution. Only an explicit `allow=True` verdict permits
    tool execution. All other cases (deny, malformed verdict, verifier exception)
    fail closed and return a synthetic ToolMessage instead of executing the tool.

    Verifier errors fail closed - exceptions from the verifier become blocked
    ToolMessages rather than propagating or allowing tool execution.

    The original `tool_call_id` is preserved verbatim in all ToolMessages.
    Tool calls are identified by their `tool_call["id"]`, not by name/args equality.

    Sync and async verifiers are supported. A synchronous verifier works on both
    sync and async paths. An async-only verifier raises RuntimeError on the sync
    path (following existing LangChain middleware conventions).

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import ToolVerifierMiddleware


        def my_verifier(request):
            # Custom verification logic
            return {
                "allow": True,
                "reason": "Verified",
                "evaluatedAt": "2026-09-22T10:00:00Z",
            }


        agent = create_agent(
            model=model,
            tools=[...],
            middleware=[ToolVerifierMiddleware(my_verifier)],
        )
        ```
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        verifier: Callable[[ToolCallRequest], dict[str, Any]]
        | Callable[[ToolCallRequest], Awaitable[dict[str, Any]]],
    ) -> None:
        """Initialize ToolVerifierMiddleware.

        Args:
            verifier: Callable that takes ToolCallRequest and returns a verdict dict.
                Must include keys:
                    - "allow": bool - whether to allow execution
                    - "reason": str - human-readable reason for decision
                    - "evaluatedAt": str - ISO timestamp when verdict was issued
                Optional keys:
                    - "expires_at": str - ISO timestamp when verdict expires
                    - "toolDefinitionHash": str - hash of tool definition for mutation detection

                Can be sync or async. Async verifiers raise RuntimeError on sync path.
        """
        super().__init__()
        self.verifier = verifier
        self._is_async = iscoroutinefunction(verifier)
        self.tools = []

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Verify tool call before execution (sync path)."""
        if self._is_async:
            msg = (
                "ToolVerifierMiddleware has an async verifier but is being used "
                "on the sync path. Use ainvoke/astream or provide a sync verifier."
            )
            raise RuntimeError(msg)

        try:
            verdict = self.verifier(request)
        except Exception:
            return self._denied_message(
                request,
                reason="Verification failed",
                evaluated_at=None,
            )

        validation_result = self._validate_verdict(request, verdict)
        if validation_result is not None:
            return validation_result

        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Verify tool call before execution (async path)."""
        try:
            if self._is_async:
                verdict = await self.verifier(request)
            else:
                verdict = self.verifier(request)
        except Exception:
            return self._denied_message(
                request,
                reason="Verification failed",
                evaluated_at=None,
            )

        validation_result = self._validate_verdict(request, verdict)
        if validation_result is not None:
            return validation_result

        return await handler(request)

    def _validate_verdict(self, request: ToolCallRequest, verdict: Any) -> ToolMessage | None:
        """Validate verdict and return denial message if invalid.

        Returns None if verdict is valid and allows execution.
        Returns ToolMessage if verdict should block execution.
        """
        # Check verdict type
        if not isinstance(verdict, dict):
            return self._denied_message(
                request,
                reason="Invalid verdict: not a dict",
                evaluated_at=None,
            )

        # Check required fields
        if "allow" not in verdict:
            return self._denied_message(
                request,
                reason="Invalid verdict: missing 'allow' field",
                evaluated_at=None,
            )

        if "reason" not in verdict:
            return self._denied_message(
                request,
                reason="Invalid verdict: missing 'reason' field",
                evaluated_at=None,
            )

        if "evaluatedAt" not in verdict:
            return self._denied_message(
                request,
                reason="Invalid verdict: missing 'evaluatedAt' field",
                evaluated_at=None,
            )

        # Check reason field type
        if not isinstance(verdict["reason"], str):
            return self._denied_message(
                request,
                reason="Invalid verdict: 'reason' must be a string",
                evaluated_at=verdict.get("evaluatedAt"),
            )

        # Check allow field type
        if not isinstance(verdict["allow"], bool):
            return self._denied_message(
                request,
                reason="Invalid verdict: 'allow' must be a boolean",
                evaluated_at=verdict.get("evaluatedAt"),
            )

        # Check evaluatedAt format
        evaluated_at_str = verdict["evaluatedAt"]
        if not isinstance(evaluated_at_str, str):
            return self._denied_message(
                request,
                reason="Invalid verdict: 'evaluatedAt' must be a string",
                evaluated_at=None,
            )

        try:
            evaluated_at = datetime.fromisoformat(evaluated_at_str.replace("Z", "+00:00"))
            if evaluated_at.tzinfo is None:
                evaluated_at = evaluated_at.replace(tzinfo=timezone.utc)
        except ValueError:
            return self._denied_message(
                request,
                reason="Invalid verdict: 'evaluatedAt' is not a valid ISO timestamp",
                evaluated_at=None,
            )

        # Check expiry if expires_at is present
        # The verifier controls the evaluation clock by setting both evaluatedAt and expires_at.
        # The middleware validates the verdict's temporal coherence and current validity:
        #
        # 1. expires_at > evaluatedAt: Verifier issued a coherent verdict (expiry after evaluation)
        # 2. evaluatedAt < now: Verdict was issued in the past (not a future-dated verdict)
        # 3. now < expires_at: Verdict is still valid at dispatch time (not yet expired)
        #
        # This follows the reference implementation (ATC v3) which checks expiry against
        # current time, while ensuring the verifier's evaluation clock is respected through
        # the coherence checks.
        if "expires_at" in verdict:
            expires_at_str = verdict["expires_at"]
            if not isinstance(expires_at_str, str):
                return self._denied_message(
                    request,
                    reason="Invalid verdict: 'expires_at' must be a string",
                    evaluated_at=evaluated_at_str,
                )
            try:
                expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
            except ValueError:
                return self._denied_message(
                    request,
                    reason="Invalid verdict: 'expires_at' is not a valid ISO timestamp",
                    evaluated_at=evaluated_at_str,
                )

            # Pinned-clock check 1: expires_at must be after evaluatedAt
            if expires_at <= evaluated_at:
                return self._denied_message(
                    request,
                    reason="Invalid verdict: expires_at must be after evaluatedAt",
                    evaluated_at=evaluated_at_str,
                )

            # Pinned-clock check 2: evaluatedAt must be in the past
            now = datetime.now(timezone.utc)
            if evaluated_at >= now:
                return self._denied_message(
                    request,
                    reason="Invalid verdict: evaluatedAt must be in the past",
                    evaluated_at=evaluated_at_str,
                )

            # Pinned-clock check 3: expires_at must be in the future (not yet expired)
            if now >= expires_at:
                return self._denied_message(
                    request,
                    reason="Verdict expired",
                    evaluated_at=evaluated_at_str,
                )

        # Check tool definition mutation if toolDefinitionHash is present
        if "toolDefinitionHash" in verdict:
            stored_hash = verdict["toolDefinitionHash"]
            if not isinstance(stored_hash, str):
                return self._denied_message(
                    request,
                    reason="Invalid verdict: 'toolDefinitionHash' must be a string",
                    evaluated_at=evaluated_at_str,
                )
            try:
                current_hash = self._compute_tool_definition_hash(request)
            except Exception:
                # If we cannot compute the hash, fail closed
                return self._denied_message(
                    request,
                    reason="Failed to compute tool definition hash",
                    evaluated_at=evaluated_at_str,
                )
            if current_hash != stored_hash:
                return self._denied_message(
                    request,
                    reason="Tool definition mutated since verification",
                    evaluated_at=evaluated_at_str,
                )

        # Check allow field value
        if not verdict["allow"]:
            return self._denied_message(
                request,
                reason=verdict["reason"],
                evaluated_at=evaluated_at_str,
            )

        # All checks passed
        return None

    def _denied_message(
        self,
        request: ToolCallRequest,
        reason: str,
        evaluated_at: str | None,
    ) -> ToolMessage:
        """Create a ToolMessage for a denied tool call.

        Args:
            request: Tool call request
            reason: Human-readable reason for denial
            evaluated_at: ISO timestamp when verdict was issued (if available)

        Returns:
            ToolMessage with error status
        """
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        content = f"Access denied: {reason}"
        if evaluated_at is not None:
            content += f" (evaluated at {evaluated_at})"

        return ToolMessage(
            content=content,
            tool_call_id=request.tool_call["id"],
            name=tool_name,
            status="error",
        )

    def _compute_tool_definition_hash(self, request: ToolCallRequest) -> str:
        """Compute a hash of the tool definition for mutation detection.

        The hash includes the tool's name, description, and argument schema.
        For unknown tools (tool=None), the hash is based on tool_call only.

        Args:
            request: Tool call request

        Returns:
            SHA256 hash as hex string with "sha256:" prefix
        """
        import hashlib
        import json

        if request.tool is None:
            # Unknown tool - hash based on tool_call
            definition = {
                "name": request.tool_call.get("name"),
                "args": request.tool_call.get("args"),
            }
        else:
            # Known tool - hash based on name, description, and args_schema
            # Use LangChain's canonical schema serialization
            args_schema_repr = self._serialize_args_schema(request.tool.args_schema)

            definition = {
                "name": request.tool.name,
                "description": request.tool.description,
                "args_schema": args_schema_repr,
            }

        # Canonical JSON serialization
        definition_str = json.dumps(definition, sort_keys=True, separators=(",", ":"))
        hash_bytes = hashlib.sha256(definition_str.encode("utf-8")).digest()
        return f"sha256:{hash_bytes.hex()}"

    def _serialize_args_schema(self, args_schema: Any) -> Any:
        """Serialize args_schema to a canonical JSON-serializable representation.

        Uses LangChain's canonical serialization utilities for deterministic hashing.
        Handles Pydantic models (v1 and v2) and dict schemas.

        Args:
            args_schema: The tool's args_schema (Pydantic model class or dict)

        Returns:
            JSON-serializable representation of the schema (dict or None)

        Raises:
            ValueError: If schema cannot be deterministically serialized
        """
        if args_schema is None:
            return None

        if isinstance(args_schema, dict):
            # Already a dict schema - return as-is
            return args_schema

        # Try to use LangChain's _serialize_args_schema if available
        try:
            from langchain_core.tools.structured import (
                _serialize_args_schema as langchain_serialize,
            )

            result = langchain_serialize(args_schema)
            # LangChain's utility may fall back to str() for unsupported schemas.
            # We only accept dict results for deterministic hashing.
            if isinstance(result, dict):
                return result
            # If result is a string, it's the non-deterministic fallback - fail closed
        except ImportError:
            pass

        # Fallback: use LangChain's model_json_schema utility
        try:
            from langchain_core.utils.pydantic import is_basemodel_subclass, model_json_schema

            if is_basemodel_subclass(args_schema):
                return model_json_schema(args_schema)
        except ImportError:
            pass

        # If we cannot serialize deterministically, fail closed
        raise ValueError(
            f"Cannot deterministically serialize args_schema for hashing. "
            f"Schema type: {type(args_schema).__name__}. "
            f"Ensure the tool's args_schema is a Pydantic model or dict."
        )

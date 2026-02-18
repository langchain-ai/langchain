"""Provenance-based execution guard for side-effecting tools.

Implements the STTI-001 invariant: "No tool with side effects should execute
unless every argument can be traced to a value produced by a prior trusted
tool output in the same session."

See: https://github.com/langchain-ai/langchain/issues/34469
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.typing import ContextT
from typing_extensions import override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ResponseT,
    ToolCallRequest,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.types import Command

logger = logging.getLogger(__name__)

PROVENANCE_ERROR_TEMPLATE = (
    "Execution blocked by provenance guard: tool '{tool_name}' has side effects "
    "but the following arguments could not be traced to any prior tool output or "
    "user input in this session:\n{details}\n"
    "Please use a read/query tool first to obtain the correct values, then retry."
)


def _build_provenance_error(tool_name: str, ungrounded: list[str]) -> str:
    """Build an error message for ungrounded arguments.

    Args:
        tool_name: The name of the tool that was blocked.
        ungrounded: List of ``"key=value"`` strings for ungrounded arguments.

    Returns:
        A formatted error message suitable for a ``ToolMessage``.
    """
    details = "\n".join(f"  - {item}" for item in ungrounded)
    return PROVENANCE_ERROR_TEMPLATE.format(tool_name=tool_name, details=details)


class ProvenanceMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Enforce argument provenance for side-effecting tools.

    Tracks values from trusted tool outputs and user inputs during a session.
    Side-effecting tools (``side_effects=True``) can only execute if their
    arguments are grounded in previously observed values. If provenance cannot
    be established, execution is blocked and an error ``ToolMessage`` is
    returned.

    This middleware is **optional and disabled by default**. Existing agents
    are unaffected unless this middleware is explicitly added.

    Configuration:
        - ``include_user_inputs``: Whether ``HumanMessage`` content counts as
            trusted provenance (default ``True``).
        - ``min_value_length``: Minimum string length for a value to require
            provenance checking. Values shorter than this are considered
            trivially common and skipped (default ``3``).

    Examples:
        !!! example "Basic usage"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ProvenanceMiddleware
            from langchain.tools import tool

            @tool
            def search(query: str) -> str:
                \"\"\"Search for information.\"\"\"
                return f"Found record: id=abc-123"

            @tool(side_effects=True)
            def delete_record(record_id: str) -> str:
                \"\"\"Delete a record by ID.\"\"\"
                return f"Deleted {record_id}"

            agent = create_agent(
                "openai:gpt-4o",
                tools=[search, delete_record],
                middleware=[ProvenanceMiddleware()],
            )
            ```

        !!! example "Strict mode (no user inputs as provenance)"

            ```python
            guard = ProvenanceMiddleware(include_user_inputs=False)
            agent = create_agent("openai:gpt-4o", tools=[...], middleware=[guard])
            ```
    """

    def __init__(
        self,
        *,
        include_user_inputs: bool = True,
        min_value_length: int = 3,
    ) -> None:
        """Initialize the provenance middleware.

        Args:
            include_user_inputs: Whether ``HumanMessage`` content counts as
                trusted provenance. When ``True`` (default), values provided
                by the user are treated as trusted.
            min_value_length: Minimum string length for a value to require
                provenance. Values shorter than this threshold are considered
                trivially common (e.g., ``"ok"``, ``"1"``) and are not checked.
        """
        super().__init__()
        self.include_user_inputs = include_user_inputs
        self.min_value_length = min_value_length

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Check argument provenance before executing side-effecting tools.

        Non-side-effecting tools are passed through without any checks.
        Side-effecting tools have their arguments validated against the
        accumulated provenance from the session's message history.

        Args:
            request: The tool call request containing the tool, arguments,
                and agent state.
            handler: The next handler in the middleware chain.

        Returns:
            A ``ToolMessage`` (either from successful execution or an error
            if provenance check fails) or a ``Command``.
        """
        tool = request.tool
        if tool is None or not getattr(tool, "side_effects", False):
            return handler(request)

        messages = request.state.get("messages", [])
        trusted_texts = self._collect_trusted_texts(messages)

        ungrounded = self._find_ungrounded_args(request.tool_call.get("args", {}), trusted_texts)

        if ungrounded:
            tool_name = request.tool_call.get("name", "unknown")
            logger.warning(
                "Provenance check failed for tool '%s': ungrounded args %s",
                tool_name,
                ungrounded,
            )
            return ToolMessage(
                content=_build_provenance_error(tool_name, ungrounded),
                tool_call_id=request.tool_call.get("id", ""),
                name=tool_name,
                status="error",
            )

        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command[Any]:
        """Async version of ``wrap_tool_call``.

        The provenance check itself is synchronous (string matching on message
        history), so this delegates to the same logic and only awaits the
        handler if the check passes.

        Args:
            request: The tool call request.
            handler: The next async handler in the middleware chain.

        Returns:
            A ``ToolMessage`` or ``Command``.
        """
        tool = request.tool
        if tool is None or not getattr(tool, "side_effects", False):
            return await handler(request)

        messages = request.state.get("messages", [])
        trusted_texts = self._collect_trusted_texts(messages)

        ungrounded = self._find_ungrounded_args(request.tool_call.get("args", {}), trusted_texts)

        if ungrounded:
            tool_name = request.tool_call.get("name", "unknown")
            logger.warning(
                "Provenance check failed for tool '%s': ungrounded args %s",
                tool_name,
                ungrounded,
            )
            return ToolMessage(
                content=_build_provenance_error(tool_name, ungrounded),
                tool_call_id=request.tool_call.get("id", ""),
                name=tool_name,
                status="error",
            )

        return await handler(request)

    def _collect_trusted_texts(self, messages: list[Any]) -> list[str]:
        """Extract text content from trusted message sources.

        Trusted sources are:
        - ``ToolMessage`` with ``status="success"`` (tool outputs)
        - ``HumanMessage`` (user inputs, if ``include_user_inputs`` is True)

        Args:
            messages: The agent's message history.

        Returns:
            List of trusted text strings.
        """
        texts: list[str] = []
        for msg in messages:
            if (isinstance(msg, ToolMessage) and msg.status == "success") or (
                self.include_user_inputs and isinstance(msg, HumanMessage)
            ):
                self._extract_text(msg.content, texts)
        return texts

    @staticmethod
    def _extract_text(content: str | list[Any], out: list[str]) -> None:
        """Extract plain text from message content.

        Handles both string content and structured content block lists.

        Args:
            content: The message content (string or list of blocks).
            out: Output list to append extracted text to.
        """
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    out.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        out.append(text)

    def _find_ungrounded_args(self, args: dict[str, Any], trusted_texts: list[str]) -> list[str]:
        """Check each argument value for provenance in trusted texts.

        Args:
            args: The tool call arguments dict.
            trusted_texts: List of trusted text strings to search.

        Returns:
            List of ``"key=value"`` strings for arguments that lack provenance.
        """
        if not trusted_texts:
            # No trusted text at all — every non-trivial argument is ungrounded
            return [
                f"{key}={value!r}"
                for key, value in args.items()
                if not self._should_skip_value(value, self.min_value_length)
            ]

        combined = "\n".join(trusted_texts)
        ungrounded: list[str] = []
        for key, value in args.items():
            if self._should_skip_value(value, self.min_value_length):
                continue
            str_value = str(value)
            if str_value not in combined:
                ungrounded.append(f"{key}={value!r}")
        return ungrounded

    @staticmethod
    def _should_skip_value(value: Any, min_length: int) -> bool:
        """Determine if a value is trivial and should skip provenance checking.

        Trivial values include: None, booleans, empty strings, and strings
        shorter than ``min_length``.

        Args:
            value: The argument value to check.
            min_length: Minimum string length to require provenance.

        Returns:
            ``True`` if the value should be skipped (not checked).
        """
        if value is None:
            return True
        if isinstance(value, bool):
            return True
        str_value = str(value)
        return len(str_value) < min_length

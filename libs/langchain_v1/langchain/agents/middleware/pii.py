"""PII detection and handling middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from typing_extensions import override

from langchain.agents.middleware._redaction import (
    PIIDetectionError,
    PIIMatch,
    RedactionRule,
    ResolvedRedactionRule,
    apply_strategy,
    detect_credit_card,
    detect_email,
    detect_ip,
    detect_mac_address,
    detect_url,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.runtime import Runtime


class PIIMiddleware(AgentMiddleware):
    """Detect and handle Personally Identifiable Information (PII) in conversations.

    This middleware detects common PII types and applies configurable strategies
    to handle them. It can detect emails, credit cards, IP addresses, MAC addresses, and
    URLs in both user input and agent output.

    Built-in PII types:

    - `email`: Email addresses
    - `credit_card`: Credit card numbers (validated with Luhn algorithm)
    - `ip`: IP addresses (validated with stdlib)
    - `mac_address`: MAC addresses
    - `url`: URLs (both `http`/`https` and bare URLs)

    Strategies:

    - `block`: Raise an exception when PII is detected
    - `redact`: Replace PII with `[REDACTED_TYPE]` placeholders
    - `mask`: Partially mask PII (e.g., `****-****-****-1234` for credit card)
    - `hash`: Replace PII with deterministic hash (e.g., `<email_hash:a1b2c3d4>`)

    Strategy Selection Guide:

    | Strategy | Preserves Identity? | Best For                                |
    | -------- | ------------------- | --------------------------------------- |
    | `block`  | N/A                 | Avoid PII completely                    |
    | `redact` | No                  | General compliance, log sanitization    |
    | `mask`   | No                  | Human readability, customer service UIs |
    | `hash`   | Yes (pseudonymous)  | Analytics, debugging                    |

    Example:
        ```python
        from langchain.agents.middleware import PIIMiddleware
        from langchain.agents import create_agent

        # Redact all emails in user input
        agent = create_agent(
            "openai:gpt-5",
            middleware=[
                PIIMiddleware("email", strategy="redact"),
            ],
        )

        # Use different strategies for different PII types
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                PIIMiddleware("credit_card", strategy="mask"),
                PIIMiddleware("url", strategy="redact"),
                PIIMiddleware("ip", strategy="hash"),
            ],
        )

        # Custom PII type with regex
        agent = create_agent(
            "openai:gpt-5",
            middleware=[
                PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
            ],
        )
        ```
    """

    def __init__(
        self,
        # From a typing point of view, the literals are covered by 'str'.
        # Nonetheless, we escape PYI051 to keep hints and autocompletion for the caller.
        pii_type: Literal["email", "credit_card", "ip", "mac_address", "url"] | str,  # noqa: PYI051
        *,
        strategy: Literal["block", "redact", "mask", "hash"] = "redact",
        detector: Callable[[str], list[PIIMatch]] | str | None = None,
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        apply_to_tool_results: bool = False,
    ) -> None:
        """Initialize the PII detection middleware.

        Args:
            pii_type: Type of PII to detect.

                Can be a built-in type (`email`, `credit_card`, `ip`, `mac_address`,
                `url`) or a custom type name.
            strategy: How to handle detected PII.

                Options:

                * `block`: Raise `PIIDetectionError` when PII is detected
                * `redact`: Replace with `[REDACTED_TYPE]` placeholders
                * `mask`: Partially mask PII (show last few characters)
                * `hash`: Replace with deterministic hash (format: `<type_hash:digest>`)

            detector: Custom detector function or regex pattern.

                * If `Callable`: Function that takes content string and returns
                    list of `PIIMatch` objects
                * If `str`: Regex pattern to match PII
                * If `None`: Uses built-in detector for the `pii_type`
            apply_to_input: Whether to check user messages before model call.
            apply_to_output: Whether to check AI messages after model call.
            apply_to_tool_results: Whether to check tool result messages after tool execution.

        Raises:
            ValueError: If `pii_type` is not built-in and no detector is provided.
        """
        super().__init__()

        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self.apply_to_tool_results = apply_to_tool_results

        self._resolved_rule: ResolvedRedactionRule = RedactionRule(
            pii_type=pii_type,
            strategy=strategy,
            detector=detector,
        ).resolve()
        self.pii_type = self._resolved_rule.pii_type
        self.strategy = self._resolved_rule.strategy
        self.detector = self._resolved_rule.detector

    @property
    def name(self) -> str:
        """Name of the middleware."""
        return f"{self.__class__.__name__}[{self.pii_type}]"

    def _process_content(self, content: str) -> tuple[str, list[PIIMatch]]:
        """Apply the configured redaction rule to the provided content."""
        matches = self.detector(content)
        if not matches:
            return content, []
        sanitized = apply_strategy(content, matches, self.strategy)
        return sanitized, matches

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Check user messages and tool results for PII before model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or `None` if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        """
        if not self.apply_to_input and not self.apply_to_tool_results:
            return None

        messages = state["messages"]
        if not messages:
            return None

        new_messages = list(messages)
        any_modified = False

        # Check user input if enabled
        if self.apply_to_input:
            # Get last user message
            last_user_msg = None
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    last_user_msg = messages[i]
                    last_user_idx = i
                    break

            if last_user_idx is not None and last_user_msg and last_user_msg.content:
                # Detect PII in message content
                content = str(last_user_msg.content)
                new_content, matches = self._process_content(content)

                if matches:
                    updated_message: AnyMessage = HumanMessage(
                        content=new_content,
                        id=last_user_msg.id,
                        name=last_user_msg.name,
                    )

                    new_messages[last_user_idx] = updated_message
                    any_modified = True

        # Check tool results if enabled
        if self.apply_to_tool_results:
            # Find the last AIMessage, then process all `ToolMessage` objects after it
            last_ai_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    last_ai_idx = i
                    break

            if last_ai_idx is not None:
                # Get all tool messages after the last AI message
                for i in range(last_ai_idx + 1, len(messages)):
                    msg = messages[i]
                    if isinstance(msg, ToolMessage):
                        tool_msg = msg
                        if not tool_msg.content:
                            continue

                        content = str(tool_msg.content)
                        new_content, matches = self._process_content(content)

                        if not matches:
                            continue

                        # Create updated tool message
                        updated_message = ToolMessage(
                            content=new_content,
                            id=tool_msg.id,
                            name=tool_msg.name,
                            tool_call_id=tool_msg.tool_call_id,
                        )

                        new_messages[i] = updated_message
                        any_modified = True

        if any_modified:
            return {"messages": new_messages}

        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async check user messages and tool results for PII before model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or `None` if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        """
        return self.before_model(state, runtime)

    @override
    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Check AI messages for PII after model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or None if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        """
        if not self.apply_to_output:
            return None

        messages = state["messages"]
        if not messages:
            return None

        # Get last AI message
        last_ai_msg = None
        last_ai_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                last_ai_msg = msg
                last_ai_idx = i
                break

        if last_ai_idx is None or not last_ai_msg or not last_ai_msg.content:
            return None

        # Detect PII in message content
        content = str(last_ai_msg.content)
        new_content, matches = self._process_content(content)

        if not matches:
            return None

        # Create updated message
        updated_message = AIMessage(
            content=new_content,
            id=last_ai_msg.id,
            name=last_ai_msg.name,
            tool_calls=last_ai_msg.tool_calls,
        )

        # Return updated messages
        new_messages = list(messages)
        new_messages[last_ai_idx] = updated_message

        return {"messages": new_messages}

    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async check AI messages for PII after model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or None if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        """
        return self.after_model(state, runtime)


__all__ = [
    "PIIDetectionError",
    "PIIMatch",
    "PIIMiddleware",
    "detect_credit_card",
    "detect_email",
    "detect_ip",
    "detect_mac_address",
    "detect_url",
]

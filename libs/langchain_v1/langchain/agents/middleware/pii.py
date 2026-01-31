"""PII detection and handling middleware for agents."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
    from collections.abc import Awaitable, Callable

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


class AsyncAnonymizerMiddleware(AgentMiddleware):
    """Anonymize PII in conversations using an external async anonymizer.

    This middleware delegates PII detection and anonymization to an external async
    function. Unlike `PIIMiddleware` which uses built-in regex-based detection and
    strategies, this middleware allows integration with external PII services that
    handle both detection and transformation.

    !!! warning "Async-only"
        This middleware requires async execution. Use it with async agent invocation
        methods. The sync hooks (`before_model`, `after_model`) will raise
        `RuntimeError` if called.

    Example:
        ```python
        from langchain.agents.middleware import AsyncAnonymizerMiddleware
        from langchain.agents import create_agent


        async def custom_anonymizer(content: str) -> str:
            # Custom anonymization logic here
            ...


        agent = create_agent(
            "my-model-id",
            middleware=[
                AsyncAnonymizerMiddleware(custom_anonymizer),
            ],
        )
        ```
    """

    def __init__(
        self,
        anonymizer: Callable[[str], Awaitable[str]],
        *,
        message_keys: str | list[str] = "messages",
        message_types: list[Literal["system", "human", "ai", "tool"]] | None = None,
    ) -> None:
        """Initialize the anonymizer middleware.

        Args:
            anonymizer: Async function that anonymizes content.

                An async callable that takes a content string and returns the
                anonymized string. The function is responsible for both detecting
                and transforming PII in a single pass.

                This is useful for integrating with external PII services
                and custom solutions.
            message_keys: The keys of the messages to anonymize.
            message_types: The types of messages to anonymize. If `None`, all
                message kinds are anonymized.

        Example:
            ```python
            async def custom_anonymizer(content: str) -> str:
                # Custom anonymization logic here
                ...


            middleware = AsyncAnonymizerMiddleware(custom_anonymizer)

            agent = create_agent(
                "my-model-id",
                middleware=[
                    AsyncAnonymizerMiddleware(custom_anonymizer),
                ],
            )
            ```
        """
        super().__init__()

        self.anonymizer = anonymizer
        self.message_types = message_types
        self.apply_to_all_message_types = message_types is None
        self.message_keys: list[str] = (
            message_keys if isinstance(message_keys, list) else [message_keys]
        )

    @property
    def name(self) -> str:
        """Name of the middleware."""
        return self.__class__.__name__

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Sync hook is not supported for async anonymizer.

        Raises:
            RuntimeError: Always raised since this middleware requires async execution.
        """
        msg = (
            f"{type(self).__name__} requires async execution. "
            "Use the async agent invocation methods instead."
        )
        raise RuntimeError(msg)

    @override
    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Sync hook is not supported for async anonymizer.

        Raises:
            RuntimeError: Always raised since this middleware requires async execution.
        """
        msg = (
            f"{type(self).__name__} requires async execution. "
            "Use the async agent invocation methods instead."
        )
        raise RuntimeError(msg)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: AgentState[Any],
        _runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Anonymize messages before model invocation.

        Args:
            state: The current agent state.
            _runtime: The langgraph runtime.

        Returns:
            Updated state with anonymized content, or `None` if no changes made.

        Raises:
            Exception: Re-raises any exception from the anonymizer coroutines.
        """
        state_update: dict[str, Any] = {}
        for message_key in self.message_keys:
            messages = state.get(message_key)
            if messages is None:
                continue
            anonymized_messages = await self._aanonymize_messages(
                cast("list[AnyMessage]", messages)
            )
            if anonymized_messages:
                state_update[message_key] = anonymized_messages
        return state_update or None

    async def aafter_model(
        self,
        state: AgentState[Any],
        _runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Anonymize messages after model invocation.

        Args:
            state: The current agent state.
            _runtime: The langgraph runtime.

        Returns:
            Updated state with anonymized content, or `None` if no changes made.

        Raises:
            Exception: Re-raises any exception from the anonymizer coroutines.
        """
        state_update: dict[str, Any] = {}
        for message_key in self.message_keys:
            messages = state.get(message_key)
            if messages is None:
                continue
            anonymized_messages = await self._aanonymize_messages(
                cast("list[AnyMessage]", messages)
            )
            if anonymized_messages:
                state_update[message_key] = anonymized_messages
        return state_update or None

    def _should_anonymize_message(self, message: AnyMessage) -> bool:
        """Check if a message should be anonymized based on message types."""
        if self.apply_to_all_message_types:
            return True

        kind_map: dict[str, type] = {
            "system": SystemMessage,
            "human": HumanMessage,
            "ai": AIMessage,
            "tool": ToolMessage,
        }
        return any(isinstance(message, kind_map[kind]) for kind in self.message_types or [])

    async def _aanonymize_content(
        self, content: str | list[str | dict[str, Any]]
    ) -> tuple[str | list[str | dict[str, Any]], bool]:
        """Anonymize content, handling str or list[str | dict].

        Returns:
            A tuple of (new_content, was_modified).
        """
        if isinstance(content, str):
            new_content = await self.anonymizer(content)
            return new_content, new_content != content

        new_list: list[str | dict[str, Any]] = []
        any_modified = False
        for item in content:
            if isinstance(item, str):
                new_item = await self.anonymizer(item)
                new_list.append(new_item)
                if new_item != item:
                    any_modified = True
            elif item.get("type") == "text":
                new_dict: dict[str, Any] = {}
                for key, value in item.items():
                    if key == "text":
                        new_value = await self.anonymizer(value)
                        new_dict[key] = new_value
                        if new_value != value:
                            any_modified = True
                    else:
                        new_dict[key] = value
                new_list.append(new_dict)
            else:
                new_list.append(item)

        return new_list, any_modified

    async def _aanonymize_message(
        self,
        message: AnyMessage,
        index: int,
    ) -> tuple[int, AnyMessage | None]:
        """Anonymize a single message.

        Args:
            message: The message to anonymize.
            index: The index of the message in the list of messages.

        Returns:
            A tuple of the index and the anonymized message, or `None` if no changes made.
        """
        if not message.content:
            return index, None

        new_content, was_modified = await self._aanonymize_content(message.content)
        if was_modified:
            return index, message.model_copy(update={"content": new_content})
        return index, None

    async def _aanonymize_messages(self, messages: list[AnyMessage]) -> list[AnyMessage] | None:
        """Anonymize a list of messages in parallel.

        Args:
            messages: The messages to anonymize.

        Returns:
            Updated messages with anonymized content, or `None` if no changes made.

        Raises:
            Exception: Re-raises any exception from the anonymizer coroutines.
        """
        coros = [
            self._aanonymize_message(message, index)
            for index, message in enumerate(messages)
            if self._should_anonymize_message(message)
        ]
        if not coros:
            return None

        results = await asyncio.gather(*coros, return_exceptions=True)
        exceptions = [result for result in results if isinstance(result, BaseException)]
        if exceptions:
            raise exceptions[0]

        new_messages = list(messages)
        any_modified = False
        for result in results:
            if not isinstance(result, BaseException):
                index, anonymized_message = result
                if anonymized_message:
                    new_messages[index] = anonymized_message
                    any_modified = True
        return new_messages if any_modified else None


__all__ = [
    "AsyncAnonymizerMiddleware",
    "PIIDetectionError",
    "PIIMatch",
    "PIIMiddleware",
    "detect_credit_card",
    "detect_email",
    "detect_ip",
    "detect_mac_address",
    "detect_url",
]

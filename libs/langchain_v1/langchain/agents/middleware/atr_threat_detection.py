"""Agent threat detection middleware using ATR-style rules.

This middleware applies detection patterns inspired by the open Agent Threat Rules
(ATR) standard at https://github.com/Agent-Threat-Rule/agent-threat-rules to user
messages and tool result messages, blocking or flagging suspicious content before
the model is invoked.

The patterns embedded here are a small built-in subset chosen for low false-positive
risk. Production deployments should load the full rule set from the ATR repository.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from typing_extensions import override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class ThreatDetectionError(ValueError):
    """Raised when content matches an ATR threat rule and strategy is `block`."""

    def __init__(self, rule_id: str, category: str, value: str) -> None:
        """Initialize with the matched rule id, category and matched substring."""
        super().__init__(
            f"Agent threat rule {rule_id} ({category}) matched content: {value[:64]!r}"
        )
        self.rule_id = rule_id
        self.category = category
        self.value = value


# Built-in ATR-inspired rules. Each tuple is (rule_id, category, regex).
# The full rule set lives at https://github.com/Agent-Threat-Rule/agent-threat-rules
# and currently covers nine attack categories with several hundred rules.
_BUILTIN_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    # Prompt injection: explicit override directives
    (
        "ATR-2026-INJ-001",
        "prompt_injection",
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+"
            r"(?:instructions?|prompts?|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    # System prompt extraction attempts
    (
        "ATR-2026-EXF-002",
        "data_exfiltration",
        re.compile(
            r"\b(?:reveal|print|repeat|show|output)\s+"
            r"(?:your\s+)?(?:system\s+prompt|initial\s+(?:prompt|instructions))\b",
            re.IGNORECASE,
        ),
    ),
    # Indirect injection markers commonly seen in poisoned tool output
    (
        "ATR-2026-INJ-003",
        "prompt_injection",
        re.compile(
            r"<\s*(?:system|assistant|tool)\s*>|"
            r"\[\s*(?:SYSTEM|ASSISTANT|INST)\s*\]",
            re.IGNORECASE,
        ),
    ),
    # Credential exfiltration: AWS access key id pattern
    (
        "ATR-2026-EXF-004",
        "credential_theft",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    # Credential exfiltration: generic bearer / OpenAI key pattern
    (
        "ATR-2026-EXF-005",
        "credential_theft",
        re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    ),
    # Shell command injection: rm -rf / and curl|sh
    (
        "ATR-2026-RCE-006",
        "code_execution",
        re.compile(
            r"\brm\s+-rf\s+/(?:\s|$)|"
            r"\bcurl\s+[^\s|]+\s*\|\s*(?:sh|bash|zsh)\b",
            re.IGNORECASE,
        ),
    ),
    # SSRF / cloud metadata endpoint
    (
        "ATR-2026-SSRF-007",
        "ssrf",
        re.compile(r"\b169\.254\.169\.254\b|/latest/meta-data/"),
    ),
)


class ATRThreatDetectionMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Detect ATR-style agent threats in user input and tool results.

    The middleware runs a small built-in pattern set inspired by the open Agent
    Threat Rules standard. Strategies:

    - `block`: raise `ThreatDetectionError` on any match
    - `flag`: tag the matched message with detection metadata in `additional_kwargs`

    Example:
        ```python
        from langchain.agents.middleware import ATRThreatDetectionMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            "openai:gpt-4o",
            middleware=[ATRThreatDetectionMiddleware(strategy="block")],
        )
        ```

    The full ATR rule set (Apache-2.0) lives at
    https://github.com/Agent-Threat-Rule/agent-threat-rules and is used in
    production by Cisco AI Defense skill-scanner and Microsoft
    agent-governance-toolkit PolicyEvaluator.
    """

    def __init__(
        self,
        *,
        strategy: Literal["block", "flag"] = "block",
        apply_to_input: bool = True,
        apply_to_tool_results: bool = True,
        rules: tuple[tuple[str, str, re.Pattern[str]], ...] | None = None,
    ) -> None:
        """Initialize the threat detection middleware.

        Args:
            strategy: How to handle a match. `block` raises `ThreatDetectionError`,
                `flag` annotates the message in `additional_kwargs["atr_matches"]`.
            apply_to_input: Scan the latest user message before model invocation.
            apply_to_tool_results: Scan tool result messages after tool execution.
            rules: Optional custom rule tuples `(rule_id, category, compiled_regex)`
                that replace the built-in set. Defaults to the built-in patterns.
        """
        super().__init__()
        self.strategy = strategy
        self.apply_to_input = apply_to_input
        self.apply_to_tool_results = apply_to_tool_results
        self.rules = rules if rules is not None else _BUILTIN_RULES

    def _scan(self, content: str) -> list[dict[str, str]]:
        matches: list[dict[str, str]] = []
        for rule_id, category, pattern in self.rules:
            matches.extend(
                {
                    "rule_id": rule_id,
                    "category": category,
                    "value": found.group(0),
                }
                for found in pattern.finditer(content)
            )
        return matches

    def _enforce(self, matches: list[dict[str, str]]) -> None:
        if self.strategy == "block" and matches:
            first = matches[0]
            raise ThreatDetectionError(
                rule_id=first["rule_id"],
                category=first["category"],
                value=first["value"],
            )

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Scan the latest user message and any subsequent tool results."""
        if not self.apply_to_input and not self.apply_to_tool_results:
            return None

        messages = state["messages"]
        if not messages:
            return None

        new_messages: list[AnyMessage] = list(messages)
        modified = False

        if self.apply_to_input:
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, HumanMessage):
                    matches = self._scan(str(msg.content or ""))
                    if matches:
                        self._enforce(matches)
                        annotated = HumanMessage(
                            content=msg.content,
                            id=msg.id,
                            name=msg.name,
                            additional_kwargs={
                                **msg.additional_kwargs,
                                "atr_matches": matches,
                            },
                        )
                        new_messages[i] = annotated
                        modified = True
                    break

        if self.apply_to_tool_results:
            last_ai_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    last_ai_idx = i
                    break
            if last_ai_idx is not None:
                for i in range(last_ai_idx + 1, len(messages)):
                    msg = messages[i]
                    if isinstance(msg, ToolMessage):
                        matches = self._scan(str(msg.content or ""))
                        if matches:
                            self._enforce(matches)
                            annotated = ToolMessage(
                                content=msg.content,
                                id=msg.id,
                                name=msg.name,
                                tool_call_id=msg.tool_call_id,
                                additional_kwargs={
                                    **msg.additional_kwargs,
                                    "atr_matches": matches,
                                },
                            )
                            new_messages[i] = annotated
                            modified = True

        if modified:
            return {"messages": new_messages}
        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async wrapper around `before_model`."""
        return self.before_model(state, runtime)

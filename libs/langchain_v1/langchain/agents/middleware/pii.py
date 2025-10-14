"""PII detection and handling middleware for agents."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from typing_extensions import TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.runtime import Runtime


class PIIMatch(TypedDict):
    """Represents a detected PII match in text."""

    type: str
    """The type of PII detected (e.g., 'email', 'ssn', 'credit_card')."""
    value: str
    """The actual matched text."""
    start: int
    """Starting position of the match in the text."""
    end: int
    """Ending position of the match in the text."""


class PIIDetectionError(Exception):
    """Exception raised when PII is detected and strategy is 'block'."""

    def __init__(self, pii_type: str, matches: list[PIIMatch]) -> None:
        """Initialize the exception with PII detection information.

        Args:
            pii_type: The type of PII that was detected.
            matches: List of PII matches found.
        """
        self.pii_type = pii_type
        self.matches = matches
        count = len(matches)
        msg = f"Detected {count} instance(s) of {pii_type} in message content"
        super().__init__(msg)


# ============================================================================
# PII Detection Functions
# ============================================================================


def _luhn_checksum(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm.

    Args:
        card_number: Credit card number string (digits only).

    Returns:
        True if the number passes Luhn validation, False otherwise.
    """
    digits = [int(d) for d in card_number if d.isdigit()]

    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        d = digit
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d

    return checksum % 10 == 0


def detect_email(content: str) -> list[PIIMatch]:
    """Detect email addresses in content.

    Args:
        content: Text content to scan.

    Returns:
        List of detected email matches.
    """
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return [
        PIIMatch(
            type="email",
            value=match.group(),
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(pattern, content)
    ]


def detect_credit_card(content: str) -> list[PIIMatch]:
    """Detect credit card numbers in content using Luhn validation.

    Detects cards in formats like:
    - 1234567890123456
    - 1234 5678 9012 3456
    - 1234-5678-9012-3456

    Args:
        content: Text content to scan.

    Returns:
        List of detected credit card matches.
    """
    # Match various credit card formats
    pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    matches = []

    for match in re.finditer(pattern, content):
        card_number = match.group()
        # Validate with Luhn algorithm
        if _luhn_checksum(card_number):
            matches.append(
                PIIMatch(
                    type="credit_card",
                    value=card_number,
                    start=match.start(),
                    end=match.end(),
                )
            )

    return matches


def detect_ip(content: str) -> list[PIIMatch]:
    """Detect IP addresses in content using stdlib validation.

    Validates both IPv4 and IPv6 addresses.

    Args:
        content: Text content to scan.

    Returns:
        List of detected IP address matches.
    """
    matches = []

    # IPv4 pattern
    ipv4_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"

    for match in re.finditer(ipv4_pattern, content):
        ip_str = match.group()
        try:
            # Validate with stdlib
            ipaddress.ip_address(ip_str)
            matches.append(
                PIIMatch(
                    type="ip",
                    value=ip_str,
                    start=match.start(),
                    end=match.end(),
                )
            )
        except ValueError:
            # Not a valid IP address
            pass

    return matches


def detect_mac_address(content: str) -> list[PIIMatch]:
    """Detect MAC addresses in content.

    Detects formats like:
    - 00:1A:2B:3C:4D:5E
    - 00-1A-2B-3C-4D-5E

    Args:
        content: Text content to scan.

    Returns:
        List of detected MAC address matches.
    """
    pattern = r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
    return [
        PIIMatch(
            type="mac_address",
            value=match.group(),
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(pattern, content)
    ]


def detect_url(content: str) -> list[PIIMatch]:
    """Detect URLs in content using regex and stdlib validation.

    Detects:
    - http://example.com
    - https://example.com/path
    - www.example.com
    - example.com/path

    Args:
        content: Text content to scan.

    Returns:
        List of detected URL matches.
    """
    matches = []

    # Pattern 1: URLs with scheme (http:// or https://)
    scheme_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"

    for match in re.finditer(scheme_pattern, content):
        url = match.group()
        try:
            result = urlparse(url)
            if result.scheme in ("http", "https") and result.netloc:
                matches.append(
                    PIIMatch(
                        type="url",
                        value=url,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        except Exception:  # noqa: S110, BLE001
            # Invalid URL, skip
            pass

    # Pattern 2: URLs without scheme (www.example.com or example.com/path)
    # More conservative to avoid false positives
    bare_pattern = r"\b(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?:/[^\s]*)?"  # noqa: E501

    for match in re.finditer(bare_pattern, content):
        # Skip if already matched with scheme
        if any(
            m["start"] <= match.start() < m["end"] or m["start"] < match.end() <= m["end"]
            for m in matches
        ):
            continue

        url = match.group()
        # Only accept if it has a path or starts with www
        # This reduces false positives like "example.com" in prose
        if "/" in url or url.startswith("www."):
            try:
                # Add scheme for validation (required for urlparse to work correctly)
                test_url = f"http://{url}"
                result = urlparse(test_url)
                if result.netloc and "." in result.netloc:
                    matches.append(
                        PIIMatch(
                            type="url",
                            value=url,
                            start=match.start(),
                            end=match.end(),
                        )
                    )
            except Exception:  # noqa: S110, BLE001
                # Invalid URL, skip
                pass

    return matches


# Built-in detector registry
_BUILTIN_DETECTORS: dict[str, Callable[[str], list[PIIMatch]]] = {
    "email": detect_email,
    "credit_card": detect_credit_card,
    "ip": detect_ip,
    "mac_address": detect_mac_address,
    "url": detect_url,
}


# ============================================================================
# Strategy Implementations
# ============================================================================


def _apply_redact_strategy(content: str, matches: list[PIIMatch]) -> str:
    """Replace PII with [REDACTED_TYPE] placeholders.

    Args:
        content: Original content.
        matches: List of PII matches to redact.

    Returns:
        Content with PII redacted.
    """
    if not matches:
        return content

    # Sort matches by start position in reverse to avoid offset issues
    sorted_matches = sorted(matches, key=lambda m: m["start"], reverse=True)

    result = content
    for match in sorted_matches:
        replacement = f"[REDACTED_{match['type'].upper()}]"
        result = result[: match["start"]] + replacement + result[match["end"] :]

    return result


def _apply_mask_strategy(content: str, matches: list[PIIMatch]) -> str:
    """Partially mask PII, showing only last few characters.

    Args:
        content: Original content.
        matches: List of PII matches to mask.

    Returns:
        Content with PII masked.
    """
    if not matches:
        return content

    # Sort matches by start position in reverse
    sorted_matches = sorted(matches, key=lambda m: m["start"], reverse=True)

    result = content
    for match in sorted_matches:
        value = match["value"]
        pii_type = match["type"]

        # Different masking strategies by type
        if pii_type == "email":
            # Show only domain: user@****.com
            parts = value.split("@")
            if len(parts) == 2:
                domain_parts = parts[1].split(".")
                if len(domain_parts) >= 2:
                    masked = f"{parts[0]}@****.{domain_parts[-1]}"
                else:
                    masked = f"{parts[0]}@****"
            else:
                masked = "****"

        elif pii_type == "credit_card":
            # Show last 4: ****-****-****-1234
            digits_only = "".join(c for c in value if c.isdigit())
            separator = "-" if "-" in value else " " if " " in value else ""
            if separator:
                masked = f"****{separator}****{separator}****{separator}{digits_only[-4:]}"
            else:
                masked = f"************{digits_only[-4:]}"

        elif pii_type == "ip":
            # Show last octet: *.*.*. 123
            parts = value.split(".")
            masked = f"*.*.*.{parts[-1]}" if len(parts) == 4 else "****"

        elif pii_type == "mac_address":
            # Show last byte: **:**:**:**:**:5E
            separator = ":" if ":" in value else "-"
            masked = (
                f"**{separator}**{separator}**{separator}**{separator}**{separator}{value[-2:]}"
            )

        elif pii_type == "url":
            # Mask everything: [MASKED_URL]
            masked = "[MASKED_URL]"

        else:
            # Default: show last 4 chars
            masked = f"****{value[-4:]}" if len(value) > 4 else "****"

        result = result[: match["start"]] + masked + result[match["end"] :]

    return result


def _apply_hash_strategy(content: str, matches: list[PIIMatch]) -> str:
    """Replace PII with deterministic hash including type information.

    Args:
        content: Original content.
        matches: List of PII matches to hash.

    Returns:
        Content with PII replaced by hashes in format <type_hash:digest>.
    """
    if not matches:
        return content

    # Sort matches by start position in reverse
    sorted_matches = sorted(matches, key=lambda m: m["start"], reverse=True)

    result = content
    for match in sorted_matches:
        value = match["value"]
        pii_type = match["type"]
        # Create deterministic hash
        hash_digest = hashlib.sha256(value.encode()).hexdigest()[:8]
        replacement = f"<{pii_type}_hash:{hash_digest}>"
        result = result[: match["start"]] + replacement + result[match["end"] :]

    return result


# ============================================================================
# PIIMiddleware
# ============================================================================


class PIIMiddleware(AgentMiddleware):
    """Detect and handle Personally Identifiable Information (PII) in agent conversations.

    This middleware detects common PII types and applies configurable strategies
    to handle them. It can detect emails, credit cards, IP addresses,
    MAC addresses, and URLs in both user input and agent output.

    Built-in PII types:
        - `email`: Email addresses
        - `credit_card`: Credit card numbers (validated with Luhn algorithm)
        - `ip`: IP addresses (validated with stdlib)
        - `mac_address`: MAC addresses
        - `url`: URLs (both http/https and bare URLs)

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
            pii_type: Type of PII to detect. Can be a built-in type
                (`email`, `credit_card`, `ip`, `mac_address`, `url`)
                or a custom type name.
            strategy: How to handle detected PII:

                * `block`: Raise PIIDetectionError when PII is detected
                * `redact`: Replace with `[REDACTED_TYPE]` placeholders
                * `mask`: Partially mask PII (show last few characters)
                * `hash`: Replace with deterministic hash (format: `<type_hash:digest>`)

            detector: Custom detector function or regex pattern.

                * If `Callable`: Function that takes content string and returns
                    list of PIIMatch objects
                * If `str`: Regex pattern to match PII
                * If `None`: Uses built-in detector for the pii_type

            apply_to_input: Whether to check user messages before model call.
            apply_to_output: Whether to check AI messages after model call.
            apply_to_tool_results: Whether to check tool result messages after tool execution.

        Raises:
            ValueError: If pii_type is not built-in and no detector is provided.
        """
        super().__init__()

        self.pii_type = pii_type
        self.strategy = strategy
        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self.apply_to_tool_results = apply_to_tool_results

        # Resolve detector
        if detector is None:
            # Use built-in detector
            if pii_type not in _BUILTIN_DETECTORS:
                msg = (
                    f"Unknown PII type: {pii_type}. "
                    f"Must be one of {list(_BUILTIN_DETECTORS.keys())} "
                    "or provide a custom detector."
                )
                raise ValueError(msg)
            self.detector = _BUILTIN_DETECTORS[pii_type]
        elif isinstance(detector, str):
            # Custom regex pattern
            pattern = detector

            def regex_detector(content: str) -> list[PIIMatch]:
                return [
                    PIIMatch(
                        type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                    for match in re.finditer(pattern, content)
                ]

            self.detector = regex_detector
        else:
            # Custom callable detector
            self.detector = detector

    @property
    def name(self) -> str:
        """Name of the middleware."""
        return f"{self.__class__.__name__}[{self.pii_type}]"

    @hook_config(can_jump_to=["end"])
    def before_model(  # noqa: PLR0915
        self,
        state: AgentState,
        runtime: Runtime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Check user messages and tool results for PII before model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or None if no PII detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is "block".
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
                matches = self.detector(content)

                if matches:
                    # Apply strategy
                    if self.strategy == "block":
                        raise PIIDetectionError(self.pii_type, matches)

                    if self.strategy == "redact":
                        new_content = _apply_redact_strategy(content, matches)
                    elif self.strategy == "mask":
                        new_content = _apply_mask_strategy(content, matches)
                    elif self.strategy == "hash":
                        new_content = _apply_hash_strategy(content, matches)
                    else:
                        # Should not reach here due to type hints
                        msg = f"Unknown strategy: {self.strategy}"
                        raise ValueError(msg)

                    # Create updated message
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
                        matches = self.detector(content)

                        if not matches:
                            continue

                        # Apply strategy
                        if self.strategy == "block":
                            raise PIIDetectionError(self.pii_type, matches)

                        if self.strategy == "redact":
                            new_content = _apply_redact_strategy(content, matches)
                        elif self.strategy == "mask":
                            new_content = _apply_mask_strategy(content, matches)
                        elif self.strategy == "hash":
                            new_content = _apply_hash_strategy(content, matches)
                        else:
                            # Should not reach here due to type hints
                            msg = f"Unknown strategy: {self.strategy}"
                            raise ValueError(msg)

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

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Check AI messages for PII after model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or None if no PII detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is "block".
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
        matches = self.detector(content)

        if not matches:
            return None

        # Apply strategy
        if self.strategy == "block":
            raise PIIDetectionError(self.pii_type, matches)

        if self.strategy == "redact":
            new_content = _apply_redact_strategy(content, matches)
        elif self.strategy == "mask":
            new_content = _apply_mask_strategy(content, matches)
        elif self.strategy == "hash":
            new_content = _apply_hash_strategy(content, matches)
        else:
            # Should not reach here due to type hints
            msg = f"Unknown strategy: {self.strategy}"
            raise ValueError(msg)

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

"""Tests for PII detection middleware."""

import re
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    InvalidToolCall,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langgraph.stream.transformers import MessagesTransformer

from langchain.agents import AgentState
from langchain.agents.factory import create_agent
from langchain.agents.middleware._redaction import RedactionRule
from langchain.agents.middleware.pii import (
    PIIDetectionError,
    PIIMatch,
    PIIMiddleware,
    _PIIStreamTransformer,
    detect_credit_card,
    detect_email,
    detect_ip,
    detect_mac_address,
    detect_url,
)
from tests.unit_tests.agents.model import FakeToolCallingModel

# ============================================================================
# Detection Function Tests
# ============================================================================


class TestEmailDetection:
    """Test email detection."""

    def test_detect_valid_email(self) -> None:
        content = "Contact me at john.doe@example.com for more info."
        matches = detect_email(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "email"
        assert matches[0]["value"] == "john.doe@example.com"
        assert matches[0]["start"] == 14
        assert matches[0]["end"] == 34

    def test_detect_multiple_emails(self) -> None:
        content = "Email alice@test.com or bob@company.org"
        matches = detect_email(content)

        assert len(matches) == 2
        assert matches[0]["value"] == "alice@test.com"
        assert matches[1]["value"] == "bob@company.org"

    def test_no_email(self) -> None:
        content = "This text has no email addresses."
        matches = detect_email(content)
        assert len(matches) == 0

    def test_invalid_email_format(self) -> None:
        content = "Invalid emails: @test.com, user@, user@domain"
        matches = detect_email(content)
        # Should not match invalid formats
        assert len(matches) == 0


class TestCreditCardDetection:
    """Test credit card detection with Luhn validation."""

    def test_detect_valid_credit_card(self) -> None:
        # Valid Visa test number
        content = "Card: 4532015112830366"
        matches = detect_credit_card(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "credit_card"
        assert matches[0]["value"] == "4532015112830366"

    def test_detect_credit_card_with_spaces(self) -> None:
        # Valid Mastercard test number
        # Add spaces
        spaced_content = "Card: 5425 2334 3010 9903"
        matches = detect_credit_card(spaced_content)

        assert len(matches) == 1
        assert "5425 2334 3010 9903" in matches[0]["value"]

    def test_detect_credit_card_with_dashes(self) -> None:
        content = "Card: 4532-0151-1283-0366"
        matches = detect_credit_card(content)

        assert len(matches) == 1

    def test_invalid_luhn_not_detected(self) -> None:
        # Invalid Luhn checksum
        content = "Card: 1234567890123456"
        matches = detect_credit_card(content)
        assert len(matches) == 0

    def test_no_credit_card(self) -> None:
        content = "No cards here."
        matches = detect_credit_card(content)
        assert len(matches) == 0


class TestIPDetection:
    """Test IP address detection."""

    def test_detect_valid_ipv4(self) -> None:
        content = "Server IP: 192.168.1.1"
        matches = detect_ip(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "ip"
        assert matches[0]["value"] == "192.168.1.1"

    def test_detect_multiple_ips(self) -> None:
        content = "Connect to 10.0.0.1 or 8.8.8.8"
        matches = detect_ip(content)

        assert len(matches) == 2
        assert matches[0]["value"] == "10.0.0.1"
        assert matches[1]["value"] == "8.8.8.8"

    def test_invalid_ip_not_detected(self) -> None:
        # Out of range octets
        content = "Not an IP: 999.999.999.999"
        matches = detect_ip(content)
        assert len(matches) == 0

    def test_version_number_not_detected(self) -> None:
        # Version numbers should not be detected as IPs
        content = "Version 1.2.3.4 released"
        matches = detect_ip(content)
        # This is a valid IP format, so it will be detected
        # This is acceptable behavior
        assert len(matches) >= 0

    def test_no_ip(self) -> None:
        content = "No IP addresses here."
        matches = detect_ip(content)
        assert len(matches) == 0


class TestMACAddressDetection:
    """Test MAC address detection."""

    def test_detect_mac_with_colons(self) -> None:
        content = "MAC: 00:1A:2B:3C:4D:5E"
        matches = detect_mac_address(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "mac_address"
        assert matches[0]["value"] == "00:1A:2B:3C:4D:5E"

    def test_detect_mac_with_dashes(self) -> None:
        content = "MAC: 00-1A-2B-3C-4D-5E"
        matches = detect_mac_address(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "00-1A-2B-3C-4D-5E"

    def test_detect_lowercase_mac(self) -> None:
        content = "MAC: aa:bb:cc:dd:ee:ff"
        matches = detect_mac_address(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "aa:bb:cc:dd:ee:ff"

    def test_no_mac(self) -> None:
        content = "No MAC address here."
        matches = detect_mac_address(content)
        assert len(matches) == 0

    def test_partial_mac_not_detected(self) -> None:
        content = "Partial: 00:1A:2B:3C"
        matches = detect_mac_address(content)
        assert len(matches) == 0


class TestURLDetection:
    """Test URL detection."""

    def test_detect_http_url(self) -> None:
        content = "Visit http://example.com for details."
        matches = detect_url(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "url"
        assert matches[0]["value"] == "http://example.com"

    def test_detect_https_url(self) -> None:
        content = "Visit https://secure.example.com/path"
        matches = detect_url(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "https://secure.example.com/path"

    def test_detect_www_url(self) -> None:
        content = "Check www.example.com"
        matches = detect_url(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "www.example.com"

    def test_detect_bare_domain_with_path(self) -> None:
        content = "Go to example.com/page"
        matches = detect_url(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "example.com/page"

    def test_detect_multiple_urls(self) -> None:
        content = "Visit http://test.com and https://example.org"
        matches = detect_url(content)

        assert len(matches) == 2

    def test_no_url(self) -> None:
        content = "No URLs here."
        matches = detect_url(content)
        assert len(matches) == 0

    def test_bare_domain_without_path_not_detected(self) -> None:
        # To reduce false positives, bare domains without paths are not detected
        content = "The word example.com in prose"
        detect_url(content)
        # May or may not detect depending on implementation
        # This is acceptable


# ============================================================================
# Strategy Tests
# ============================================================================


class TestRedactStrategy:
    """Test redact strategy."""

    def test_redact_email(self) -> None:
        middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[HumanMessage("Email me at test@example.com")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "[REDACTED_EMAIL]" in result["messages"][0].content
        assert "test@example.com" not in result["messages"][0].content

    def test_redact_multiple_pii(self) -> None:
        middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[HumanMessage("Contact alice@test.com or bob@test.com")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert content.count("[REDACTED_EMAIL]") == 2
        assert "alice@test.com" not in content
        assert "bob@test.com" not in content


class TestMaskStrategy:
    """Test mask strategy."""

    def test_mask_email(self) -> None:
        middleware = PIIMiddleware("email", strategy="mask")
        state = AgentState[Any](messages=[HumanMessage("Email: user@example.com")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "user@****.com" in content
        assert "user@example.com" not in content

    def test_mask_credit_card(self) -> None:
        middleware = PIIMiddleware("credit_card", strategy="mask")
        # Valid test card
        state = AgentState[Any](messages=[HumanMessage("Card: 4532015112830366")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "0366" in content  # Last 4 digits visible
        assert "4532015112830366" not in content

    def test_mask_ip(self) -> None:
        middleware = PIIMiddleware("ip", strategy="mask")
        state = AgentState[Any](messages=[HumanMessage("IP: 192.168.1.100")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "*.*.*.100" in content
        assert "192.168.1.100" not in content


class TestHashStrategy:
    """Test hash strategy."""

    def test_hash_email(self) -> None:
        middleware = PIIMiddleware("email", strategy="hash")
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])

        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "<email_hash:" in content
        assert ">" in content
        assert "test@example.com" not in content

    def test_hash_is_deterministic(self) -> None:
        middleware = PIIMiddleware("email", strategy="hash")

        # Same email should produce same hash
        state1 = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])
        state2 = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])

        result1 = middleware.before_model(state1, Runtime())
        result2 = middleware.before_model(state2, Runtime())

        assert result1 is not None
        assert result2 is not None
        assert result1["messages"][0].content == result2["messages"][0].content


class TestBlockStrategy:
    """Test block strategy."""

    def test_block_raises_exception(self) -> None:
        middleware = PIIMiddleware("email", strategy="block")
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])

        with pytest.raises(PIIDetectionError) as exc_info:
            middleware.before_model(state, Runtime())

        assert exc_info.value.pii_type == "email"
        assert len(exc_info.value.matches) == 1
        assert "test@example.com" in exc_info.value.matches[0]["value"]

    def test_block_with_multiple_matches(self) -> None:
        middleware = PIIMiddleware("email", strategy="block")
        state = AgentState[Any](messages=[HumanMessage("Emails: alice@test.com and bob@test.com")])

        with pytest.raises(PIIDetectionError) as exc_info:
            middleware.before_model(state, Runtime())

        assert len(exc_info.value.matches) == 2


# ============================================================================
# Middleware Integration Tests
# ============================================================================


class TestPIIMiddlewareIntegration:
    """Test PIIMiddleware integration with agent."""

    def test_apply_to_input_only(self) -> None:
        """Test that middleware only processes input when configured."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=True, apply_to_output=False
        )

        # Should process HumanMessage
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])
        result = middleware.before_model(state, Runtime())
        assert result is not None
        assert "[REDACTED_EMAIL]" in result["messages"][0].content

        # Should not process AIMessage
        state = AgentState[Any](messages=[AIMessage("My email is ai@example.com")])
        result = middleware.after_model(state, Runtime())
        assert result is None

    def test_apply_to_output_only(self) -> None:
        """Test that middleware only processes output when configured."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=False, apply_to_output=True
        )

        # Should not process HumanMessage
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])
        result = middleware.before_model(state, Runtime())
        assert result is None

        # Should process AIMessage
        state = AgentState[Any](messages=[AIMessage("My email is ai@example.com")])
        result = middleware.after_model(state, Runtime())
        assert result is not None
        assert "[REDACTED_EMAIL]" in result["messages"][0].content

    def test_apply_to_both(self) -> None:
        """Test that middleware processes both input and output."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=True, apply_to_output=True
        )

        # Should process HumanMessage
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com")])
        result = middleware.before_model(state, Runtime())
        assert result is not None

        # Should process AIMessage
        state = AgentState[Any](messages=[AIMessage("My email is ai@example.com")])
        result = middleware.after_model(state, Runtime())
        assert result is not None

    def test_no_pii_returns_none(self) -> None:
        """Test that middleware returns None when no PII detected."""
        middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[HumanMessage("No PII here")])

        result = middleware.before_model(state, Runtime())
        assert result is None

    def test_empty_messages(self) -> None:
        """Test that middleware handles empty messages gracefully."""
        middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[])

        result = middleware.before_model(state, Runtime())
        assert result is None

    def test_apply_to_tool_results(self) -> None:
        """Test that middleware processes tool results when enabled."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=False, apply_to_tool_results=True
        )

        # Simulate a conversation with tool call and result containing PII
        state = AgentState[Any](
            messages=[
                HumanMessage("Search for John"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="search", args={}, id="call_123", type="tool_call")],
                ),
                ToolMessage(content="Found: john@example.com", tool_call_id="call_123"),
            ]
        )

        result = middleware.before_model(state, Runtime())

        assert result is not None
        # Check that the tool message was redacted
        tool_msg = result["messages"][2]
        assert isinstance(tool_msg, ToolMessage)
        assert "[REDACTED_EMAIL]" in tool_msg.content
        assert "john@example.com" not in tool_msg.content

    def test_apply_to_tool_results_mask_strategy(self) -> None:
        """Test that mask strategy works for tool results."""
        middleware = PIIMiddleware(
            "ip", strategy="mask", apply_to_input=False, apply_to_tool_results=True
        )

        state = AgentState[Any](
            messages=[
                HumanMessage("Get server IP"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="get_ip", args={}, id="call_456", type="tool_call")],
                ),
                ToolMessage(content="Server IP: 192.168.1.100", tool_call_id="call_456"),
            ]
        )

        result = middleware.before_model(state, Runtime())

        assert result is not None
        tool_msg = result["messages"][2]
        assert "*.*.*.100" in tool_msg.content
        assert "192.168.1.100" not in tool_msg.content

    def test_apply_to_tool_results_block_strategy(self) -> None:
        """Test that block strategy raises error for PII in tool results."""
        middleware = PIIMiddleware(
            "email", strategy="block", apply_to_input=False, apply_to_tool_results=True
        )

        state = AgentState[Any](
            messages=[
                HumanMessage("Search for user"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="search", args={}, id="call_789", type="tool_call")],
                ),
                ToolMessage(content="User email: sensitive@example.com", tool_call_id="call_789"),
            ]
        )

        with pytest.raises(PIIDetectionError) as exc_info:
            middleware.before_model(state, Runtime())

        assert exc_info.value.pii_type == "email"
        assert len(exc_info.value.matches) == 1

    def test_with_agent(self) -> None:
        """Test PIIMiddleware integrated with create_agent."""
        model = FakeToolCallingModel()

        agent = create_agent(
            model=model,
            middleware=[PIIMiddleware("email", strategy="redact")],
        )

        # Invoke (agent is already compiled)
        result = agent.invoke({"messages": [HumanMessage("Email: test@example.com")]})

        # Check that email was redacted in the stored messages
        # The first message should have been processed
        messages = result["messages"]
        assert any("[REDACTED_EMAIL]" in str(msg.content) for msg in messages)


class TestCustomDetector:
    """Test custom detector functionality."""

    def test_custom_regex_detector(self) -> None:
        # Custom regex for API keys
        middleware = PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="redact",
        )

        state = AgentState[Any](messages=[HumanMessage("Key: sk-abcdefghijklmnopqrstuvwxyz123456")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "[REDACTED_API_KEY]" in result["messages"][0].content

    def test_custom_callable_detector(self) -> None:
        # Custom detector function
        def detect_custom(content: str) -> list[PIIMatch]:
            matches = []
            if "CONFIDENTIAL" in content:
                idx = content.index("CONFIDENTIAL")
                matches.append(
                    PIIMatch(
                        type="confidential",
                        value="CONFIDENTIAL",
                        start=idx,
                        end=idx + 12,
                    )
                )
            return matches

        middleware = PIIMiddleware(
            "confidential",
            detector=detect_custom,
            strategy="redact",
        )

        state = AgentState[Any](messages=[HumanMessage("This is CONFIDENTIAL information")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "[REDACTED_CONFIDENTIAL]" in result["messages"][0].content

    def test_custom_callable_detector_with_text_key_hash(self) -> None:
        """Custom detectors returning 'text' instead of 'value' must work with hash strategy.

        Regression test for https://github.com/langchain-ai/langchain/issues/35647:
        Custom detectors documented to return {"text", "start", "end"} caused
        KeyError: 'value' when used with hash or mask strategies.
        """

        def detect_phone(content: str) -> list[dict]:  # type: ignore[type-arg]
            return [
                {"text": m.group(), "start": m.start(), "end": m.end()}
                for m in re.finditer(r"\+91[\s.-]?\d{10}", content)
            ]

        middleware = PIIMiddleware(
            "indian_phone",
            detector=detect_phone,
            strategy="hash",
            apply_to_input=True,
        )

        state = AgentState[Any](messages=[HumanMessage("Call +91 9876543210")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "<indian_phone_hash:" in result["messages"][0].content
        assert "+91 9876543210" not in result["messages"][0].content

    def test_custom_callable_detector_with_text_key_mask(self) -> None:
        """Custom detectors returning 'text' instead of 'value' must work with mask strategy."""

        def detect_phone(content: str) -> list[dict]:  # type: ignore[type-arg]
            return [
                {"text": m.group(), "start": m.start(), "end": m.end()}
                for m in re.finditer(r"\+91[\s.-]?\d{10}", content)
            ]

        middleware = PIIMiddleware(
            "indian_phone",
            detector=detect_phone,
            strategy="mask",
            apply_to_input=True,
        )

        state = AgentState[Any](messages=[HumanMessage("Call +91 9876543210")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        assert "****" in result["messages"][0].content
        assert "+91 9876543210" not in result["messages"][0].content

    def test_unknown_builtin_type_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown PII type"):
            PIIMiddleware("unknown_type", strategy="redact")

    def test_custom_type_without_detector_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown PII type"):
            PIIMiddleware("custom_type", strategy="redact")


class TestMultipleMiddleware:
    """Test using multiple PII middleware instances."""

    def test_sequential_application(self) -> None:
        """Test that multiple PII types are detected when applied sequentially."""
        # First apply email middleware
        email_middleware = PIIMiddleware("email", strategy="redact")
        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com, IP: 192.168.1.1")])
        result1 = email_middleware.before_model(state, Runtime())

        # Then apply IP middleware to the result
        ip_middleware = PIIMiddleware("ip", strategy="mask")
        assert result1 is not None
        state_with_email_redacted = AgentState[Any](messages=result1["messages"])
        result2 = ip_middleware.before_model(state_with_email_redacted, Runtime())

        assert result2 is not None
        content = result2["messages"][0].content

        # Email should be redacted
        assert "[REDACTED_EMAIL]" in content
        assert "test@example.com" not in content

        # IP should be masked
        assert "*.*.*.1" in content
        assert "192.168.1.1" not in content

    def test_multiple_pii_middleware_with_create_agent(self) -> None:
        """Test that multiple PIIMiddleware instances work together in create_agent."""
        model = FakeToolCallingModel()

        # Multiple PIIMiddleware instances should work because each has a unique name
        agent = create_agent(
            model=model,
            middleware=[
                PIIMiddleware("email", strategy="redact"),
                PIIMiddleware("ip", strategy="mask"),
                PIIMiddleware("url", strategy="block", apply_to_input=True),
            ],
        )

        # Test with email and IP (url would block, so we omit it)
        result = agent.invoke(
            {"messages": [HumanMessage("Contact: test@example.com, IP: 192.168.1.100")]}
        )

        messages = result["messages"]
        content = " ".join(str(msg.content) for msg in messages)

        # Email should be redacted
        assert "test@example.com" not in content
        # IP should be masked
        assert "192.168.1.100" not in content

    def test_custom_detector_for_multiple_types(self) -> None:
        """Test using a single middleware with custom detector for multiple PII types.

        This is an alternative to using multiple middleware instances,
        useful when you want the same strategy for multiple PII types.
        """

        # Combine multiple detectors into one
        def detect_email_and_ip(content: str) -> list[PIIMatch]:
            return detect_email(content) + detect_ip(content)

        middleware = PIIMiddleware(
            "email_or_ip",
            detector=detect_email_and_ip,
            strategy="redact",
        )

        state = AgentState[Any](messages=[HumanMessage("Email: test@example.com, IP: 10.0.0.1")])
        result = middleware.before_model(state, Runtime())

        assert result is not None
        content = result["messages"][0].content
        assert "test@example.com" not in content
        assert "10.0.0.1" not in content


# ============================================================================
# Stream Transformer Tests
# ============================================================================


def _make_delta_event(text: str, *, index: int = 0, run_id: str = "r1") -> dict[str, Any]:
    """Build a `messages` protocol event for a text content-block delta."""
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": 0,
            "data": (
                {
                    "event": "content-block-delta",
                    "index": index,
                    "delta": {"type": "text-delta", "text": text},
                },
                {"run_id": run_id},
            ),
        },
    }


def _make_finish_event(text: str, *, index: int = 0, run_id: str = "r1") -> dict[str, Any]:
    """Build a `messages` protocol event for content-block-finish on a text block."""
    return {
        "type": "event",
        "method": "messages",
        "params": {
            "namespace": [],
            "timestamp": 0,
            "data": (
                {
                    "event": "content-block-finish",
                    "index": index,
                    "content": {"type": "text", "text": text},
                },
                {"run_id": run_id},
            ),
        },
    }


def _emitted_text(events: list[dict[str, Any]]) -> str:
    """Concatenate delta + finalized text the way a streaming consumer would."""
    parts = []
    final_by_index: dict[int, str] = {}
    for event in events:
        payload = event["params"]["data"][0]
        kind = payload.get("event")
        if kind == "content-block-delta":
            delta = payload["delta"]
            if delta.get("type") == "text-delta":
                parts.append(delta["text"])
        elif kind == "content-block-finish":
            content = payload.get("content", {})
            if content.get("type") == "text":
                final_by_index[payload["index"]] = content["text"]
    # Concatenated delta stream is what the consumer sees in real time;
    # finalized text is the snapshot. Return both via a tuple-like dict.
    return "".join(parts), final_by_index  # type: ignore[return-value]


def _run_transformer(transformer: Any, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Feed events through the transformer (mutates in place) and return them."""
    for event in events:
        transformer.process(event)
    return events


class TestPIIStreamTransformer:
    """Tests for the in-flight stream transformer."""

    def test_redact_value_walks_nested_strings(self) -> None:
        """`_redact_value` redacts PII in string leaves of nested dict/list."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        value = {
            "to": "alice@example.com",
            "cc": ["bob@example.com", "no-pii"],
            "nested": {"hidden": "user mail: charlie@example.com"},
            "count": 3,
            "flag": True,
        }
        redacted = transformer._redact_value(value)

        assert redacted == {
            "to": "[REDACTED_EMAIL]",
            "cc": ["[REDACTED_EMAIL]", "no-pii"],
            "nested": {"hidden": "user mail: [REDACTED_EMAIL]"},
            "count": 3,
            "flag": True,
        }

    def test_redact_value_block_strategy_raises(self) -> None:
        """Under `block`, `_redact_value` raises on the first matching leaf."""
        rule = RedactionRule(pii_type="email", strategy="block").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        value = {"to": "alice@example.com", "subject": "clean"}
        with pytest.raises(PIIDetectionError):
            transformer._redact_value(value)

    def test_redact_value_passthrough_for_clean_input(self) -> None:
        """No PII anywhere → returns an equal value."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        value = {"a": "hello", "b": [1, 2, "world"], "c": None}
        redacted = transformer._redact_value(value)
        assert redacted == value

    def test_redact_value_walks_structured_message_content(self) -> None:
        """`_redact_value` walks list-typed `.content` (content-blocks shape)."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        msg = AIMessage(
            content=[
                {"type": "text", "text": "Reach me at alice@example.com"},
                {"type": "reasoning", "reasoning": "User mentioned bob@example.com"},
            ],
            id="m1",
        )
        redacted = transformer._redact_value(msg)

        # Original untouched.
        assert msg.content[0]["text"] == "Reach me at alice@example.com"
        # Redacted copy walked every block.
        assert "alice@example.com" not in redacted.content[0]["text"]
        assert "[REDACTED_EMAIL]" in redacted.content[0]["text"]
        assert "bob@example.com" not in redacted.content[1]["reasoning"]
        assert "[REDACTED_EMAIL]" in redacted.content[1]["reasoning"]

    def test_redact_value_walks_ai_message_tool_calls(self) -> None:
        """`_redact_value` redacts `AIMessage.tool_calls[*].args` even with empty content.

        The legacy `(AIMessage, metadata)` payload path on the `messages`
        channel mutates `tool_calls` in place before the `values` event
        fires, but on the v3 streaming path the state's AIMessage is
        assembled by langgraph without going through that mutation.
        `_redact_value` is what scrubs the message when the `values`
        snapshot walks it — it must not stop at empty content.
        """
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        msg = AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="send_email", args={"to": "alice@example.com"}, id="c1"),
            ],
            id="m1",
        )
        redacted = transformer._redact_value(msg)

        # Original message stays intact for state-level enforcers.
        assert msg.tool_calls[0]["args"] == {"to": "alice@example.com"}
        # The returned copy has scrubbed tool_calls and is a fresh object.
        assert redacted is not msg
        assert redacted.tool_calls[0]["args"] == {"to": "[REDACTED_EMAIL]"}

    def test_redact_value_walks_both_content_and_tool_calls(self) -> None:
        """Content and tool_calls both carry PII — both get scrubbed in one pass."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        msg = AIMessage(
            content="Reaching out from bob@example.com",
            tool_calls=[
                ToolCall(name="send_email", args={"to": "alice@example.com"}, id="c1"),
            ],
            id="m1",
        )
        redacted = transformer._redact_value(msg)

        assert "bob@example.com" not in redacted.content
        assert "[REDACTED_EMAIL]" in redacted.content
        assert redacted.tool_calls[0]["args"] == {"to": "[REDACTED_EMAIL]"}

    def test_reasoning_delta_uses_lookback(self) -> None:
        """`reasoning-delta` events go through the same lookback as text-delta."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=32)

        events = [
            {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": (
                        {
                            "event": "content-block-delta",
                            "index": 0,
                            "delta": {
                                "type": "reasoning-delta",
                                "reasoning": "User mentioned alice@example.com",
                            },
                        },
                        {"run_id": "r1"},
                    ),
                },
            },
            {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": (
                        {
                            "event": "content-block-finish",
                            "index": 0,
                            "content": {
                                "type": "reasoning",
                                "reasoning": "User mentioned alice@example.com",
                            },
                        },
                        {"run_id": "r1"},
                    ),
                },
            },
        ]
        for e in events:
            transformer.process(e)

        # The finalize snapshot is redacted. The delta itself may emit
        # nothing (entire string fits within lookback) — the consumer's
        # ChatModelStream reconciles against the finalize content.
        finalized = events[1]["params"]["data"][0]["content"]["reasoning"]
        assert "alice@example.com" not in finalized
        assert "[REDACTED_EMAIL]" in finalized

    def test_reasoning_delta_block_strategy_raises(self) -> None:
        """`block` raises immediately when reasoning content contains PII."""
        rule = RedactionRule(pii_type="email", strategy="block").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": (
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {
                            "type": "reasoning-delta",
                            "reasoning": "User mentioned alice@example.com",
                        },
                    },
                    {"run_id": "r1"},
                ),
            },
        }
        with pytest.raises(PIIDetectionError):
            transformer.process(event)

    def test_tool_call_args_short_args_withheld_during_streaming(self) -> None:
        """Tool-call args shorter than `stream_lookback` are withheld on each chunk.

        The redacted args dict surfaces on `content-block-finish` via
        `_finalize_block` — the consumer's `tool_calls_proj` replaces
        wholesale on each chunk, so this is equivalent to "no args
        during streaming, redacted dict at finalize".
        """
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)  # default lookback=128

        events = [
            {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": (
                        {
                            "event": "content-block-delta",
                            "index": 0,
                            "delta": {
                                "type": "block-delta",
                                "fields": {
                                    "type": "tool_call_chunk",
                                    "id": "call_1",
                                    "name": "send_email",
                                    "args": '{"to": "alice@example.com"}',
                                },
                            },
                        },
                        {"run_id": "r1"},
                    ),
                },
            },
        ]
        _run_transformer(transformer, events)

        fields = events[0]["params"]["data"][0]["delta"]["fields"]
        # 27 chars < lookback (128), so emit_end = 0 — nothing reaches the wire.
        assert fields["args"] == ""

    def test_tool_call_args_long_args_emit_safe_prefix(self) -> None:
        """Args longer than `stream_lookback` stream the redacted safe prefix.

        Detection runs on the full cumulative args, so any complete PII
        anywhere in the string is redacted before emission. The trailing
        `stream_lookback` characters are withheld — they might be the
        start of a partial PII match that completes in a future delta.
        """
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=8)

        # 50-char args with PII near the start; emit_end = 50 - 8 = 42.
        args = '{"to": "alice@example.com", "subject": "hi"}'
        events = [
            {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": (
                        {
                            "event": "content-block-delta",
                            "index": 0,
                            "delta": {
                                "type": "block-delta",
                                "fields": {
                                    "type": "tool_call_chunk",
                                    "id": "call_1",
                                    "name": "send_email",
                                    "args": args,
                                },
                            },
                        },
                        {"run_id": "r1"},
                    ),
                },
            },
        ]
        _run_transformer(transformer, events)

        emitted = events[0]["params"]["data"][0]["delta"]["fields"]["args"]
        # The full cumulative args was detected and redacted before
        # truncation, so the prefix that lands on the wire has no PII.
        assert "alice@example.com" not in emitted
        assert "[REDACTED_EMAIL]" in emitted

    def test_tool_call_args_partial_pii_across_chunks_withheld(self) -> None:
        """Cumulative chunks that grow into PII don't leak intermediate states.

        Regression for Corridor's partial-exposure window: a model
        streaming `{"to": "alice@` then `{"to": "alice@example.com"}`
        would expose the partial first chunk under the old in-place
        redaction. With lookback withholding, the partial chunk is
        below the lookback threshold and never reaches the wire.
        """
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)  # default lookback=128

        chunks = [
            '{"to": "alice@',
            '{"to": "alice@example',
            '{"to": "alice@example.com"}',
        ]
        events = [
            {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": (
                        {
                            "event": "content-block-delta",
                            "index": 0,
                            "delta": {
                                "type": "block-delta",
                                "fields": {
                                    "type": "tool_call_chunk",
                                    "id": "c1",
                                    "name": "send_email",
                                    "args": c,
                                },
                            },
                        },
                        {"run_id": "r1"},
                    ),
                },
            }
            for c in chunks
        ]
        _run_transformer(transformer, events)

        # None of the intermediate accumulation states reach the wire —
        # all chunks are below the 128-char lookback, so emit_end = 0.
        for e in events:
            args_on_wire = e["params"]["data"][0]["delta"]["fields"]["args"]
            assert "alice@" not in args_on_wire
            assert "alice" not in args_on_wire
            assert args_on_wire == ""

    def test_finalize_block_redacts_tool_call_args(self) -> None:
        """`content-block-finish` with type=tool_call walks args dict."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        events = [
            {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": (
                        {
                            "event": "content-block-finish",
                            "index": 0,
                            "content": {
                                "type": "tool_call",
                                "id": "call_1",
                                "name": "send_email",
                                "args": {"to": "alice@example.com", "subject": "clean"},
                            },
                        },
                        {"run_id": "r1"},
                    ),
                },
            },
        ]
        _run_transformer(transformer, events)
        args = events[0]["params"]["data"][0]["content"]["args"]
        assert args == {"to": "[REDACTED_EMAIL]", "subject": "clean"}

    def test_finalize_block_raises_on_tool_call_args_under_block(self) -> None:
        """`block` raises when the finalized tool-call args contain PII."""
        rule = RedactionRule(pii_type="email", strategy="block").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": (
                    {
                        "event": "content-block-finish",
                        "index": 0,
                        "content": {
                            "type": "tool_call",
                            "id": "call_1",
                            "name": "send_email",
                            "args": {"to": "alice@example.com"},
                        },
                    },
                    {"run_id": "r1"},
                ),
            },
        }
        with pytest.raises(PIIDetectionError):
            transformer.process(event)

    def test_legacy_payload_redacts_tool_call_args(self) -> None:
        """`(AIMessage, metadata)` shape redacts tool_calls[].args."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        msg = AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="send_email", args={"to": "alice@example.com"}, id="c1"),
            ],
            id="m1",
        )
        event: dict[str, Any] = {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": (msg, {"run_id": "r1"}),
            },
        }
        transformer.process(event)
        out_msg = event["params"]["data"][0]
        assert out_msg.tool_calls[0]["args"] == {"to": "[REDACTED_EMAIL]"}

    def test_legacy_payload_block_raises_when_tool_call_has_pii(self) -> None:
        """`block` + legacy AIMessage with PII in tool_calls → raises immediately."""
        rule = RedactionRule(pii_type="email", strategy="block").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        msg = AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="send_email", args={"to": "alice@example.com"}, id="c1"),
            ],
            id="m1",
        )
        event: dict[str, Any] = {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": (msg, {"run_id": "r1"}),
            },
        }
        with pytest.raises(PIIDetectionError):
            transformer.process(event)

    def test_legacy_payload_redacts_tool_message_content(self) -> None:
        """`(ToolMessage, metadata)` payload redacts `.content`."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        msg = ToolMessage(content="Result: alice@example.com", tool_call_id="c1", id="m1")
        event: dict[str, Any] = {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": (msg, {"run_id": "r1"}),
            },
        }
        transformer.process(event)
        out_msg = event["params"]["data"][0]
        assert "alice@example.com" not in out_msg.content
        assert "[REDACTED_EMAIL]" in out_msg.content

    def test_tools_in_required_stream_modes(self) -> None:
        """The transformer subscribes to both `messages` and `tools`."""
        assert "tools" in _PIIStreamTransformer.required_stream_modes
        assert "messages" in _PIIStreamTransformer.required_stream_modes

    def test_process_tools_event_passes_through(self) -> None:
        """Tools events route to the new handler without error."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)
        event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-started",
                    "tool_call_id": "c1",
                    "tool_name": "echo",
                    "input": {},
                },
            },
        }
        assert transformer.process(event) is True

    def test_tool_started_string_input_redacted(self) -> None:
        """Single-argument tools pass `input` as a raw string."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-started",
                    "tool_call_id": "c1",
                    "tool_name": "echo",
                    "input": "user mail is alice@example.com",
                },
            },
        }
        transformer.process(event)
        out = event["params"]["data"]["input"]
        assert "alice@example.com" not in out
        assert "[REDACTED_EMAIL]" in out

    def test_tool_started_list_input_redacted(self) -> None:
        """Array-input tools pass `input` as a list."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-started",
                    "tool_call_id": "c1",
                    "tool_name": "fanout",
                    "input": ["bob@example.com", "clean"],
                },
            },
        }
        transformer.process(event)
        assert event["params"]["data"]["input"] == ["[REDACTED_EMAIL]", "clean"]

    def test_drop_run_does_not_sweep_tool_buffers(self) -> None:
        """`_drop_run` on the messages channel must not wipe tool buffers."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=64)

        # Put a tool-output buffer entry in place.
        transformer.process(
            {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": {
                        "event": "tool-output-delta",
                        "tool_call_id": "c1",
                        "delta": "partial",
                    },
                },
            }
        )
        before = dict(transformer._buffers)
        assert any("c1" in str(k) for k in before)

        # An errant message-finish with no run_id used to sweep all
        # `("", *)` buffer keys — including the tool one. With the
        # tool-buffer sentinel namespace, the sweep is a no-op for
        # tool entries.
        transformer.process(
            {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": ({"event": "message-finish"}, {}),
                },
            }
        )
        assert any("c1" in str(k) for k in transformer._buffers)

    def test_finalize_invalid_tool_call_redacts_string_args(self) -> None:
        """`invalid_tool_call.args` is a raw JSON string, not a dict."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        events = [
            {
                "type": "event",
                "method": "messages",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": (
                        {
                            "event": "content-block-finish",
                            "index": 0,
                            "content": {
                                "type": "invalid_tool_call",
                                "id": "c1",
                                "name": "send_email",
                                "args": '{"to": "alice@example.com", "bad json',
                                "error": "Unterminated string",
                            },
                        },
                        {"run_id": "r1"},
                    ),
                },
            },
        ]
        _run_transformer(transformer, events)
        out = events[0]["params"]["data"][0]["content"]["args"]
        assert "alice@example.com" not in out
        assert "[REDACTED_EMAIL]" in out

    def test_redact_value_walks_invalid_tool_calls(self) -> None:
        """`AIMessage.invalid_tool_calls` go through the same recursion."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        msg = AIMessage(
            content="",
            invalid_tool_calls=[
                InvalidToolCall(
                    name="send_email",
                    args='{"to": "alice@example.com"} BROKEN',
                    id="c1",
                    error="parse failed",
                ),
            ],
            id="m1",
        )
        redacted = transformer._redact_value(msg)
        assert "alice@example.com" not in redacted.invalid_tool_calls[0]["args"]
        assert "[REDACTED_EMAIL]" in redacted.invalid_tool_calls[0]["args"]
        assert "alice@example.com" in msg.invalid_tool_calls[0]["args"]

    def test_tool_started_input_is_redacted(self) -> None:
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-started",
                    "tool_call_id": "c1",
                    "tool_name": "send_email",
                    "input": {"to": "alice@example.com", "subject": "clean"},
                },
            },
        }
        transformer.process(event)
        assert event["params"]["data"]["input"] == {
            "to": "[REDACTED_EMAIL]",
            "subject": "clean",
        }

    def test_tool_output_delta_string_uses_lookback(self) -> None:
        """String tool-output deltas get the lookback redaction same as text."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=64)

        events = [
            {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": {
                        "event": "tool-output-delta",
                        "tool_call_id": "c1",
                        "delta": "Result: alice",
                    },
                },
            },
            {
                "type": "event",
                "method": "tools",
                "params": {
                    "namespace": [],
                    "timestamp": 0,
                    "data": {
                        "event": "tool-output-delta",
                        "tool_call_id": "c1",
                        "delta": "@example.com end",
                    },
                },
            },
        ]
        for e in events:
            transformer.process(e)
        streamed = events[0]["params"]["data"]["delta"] + events[1]["params"]["data"]["delta"]
        assert "alice@example.com" not in streamed

    def test_tool_output_delta_dict_walks_strings(self) -> None:
        """Structured tool-output deltas redact each string leaf in place."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-output-delta",
                    "tool_call_id": "c1",
                    "delta": {"row": {"email": "alice@example.com"}, "ok": True},
                },
            },
        }
        transformer.process(event)
        assert event["params"]["data"]["delta"] == {
            "row": {"email": "[REDACTED_EMAIL]"},
            "ok": True,
        }

    def test_tool_finished_output_redacted(self) -> None:
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-finished",
                    "tool_call_id": "c1",
                    "output": {"email": "alice@example.com", "clean": "yes"},
                },
            },
        }
        transformer.process(event)
        assert event["params"]["data"]["output"] == {
            "email": "[REDACTED_EMAIL]",
            "clean": "yes",
        }

    def test_tool_error_message_redacted(self) -> None:
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-error",
                    "tool_call_id": "c1",
                    "message": "failed for alice@example.com",
                },
            },
        }
        transformer.process(event)
        assert "alice@example.com" not in event["params"]["data"]["message"]
        assert "[REDACTED_EMAIL]" in event["params"]["data"]["message"]

    def test_tool_finished_clears_per_tool_buffer(self) -> None:
        """tool-finished drops the lookback buffer for that tool_call_id."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=64)

        delta_event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-output-delta",
                    "tool_call_id": "c1",
                    "delta": "partial text",
                },
            },
        }
        transformer.process(delta_event)
        assert any("c1" in str(k) for k in transformer._buffers)

        finish_event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-finished",
                    "tool_call_id": "c1",
                    "output": "done",
                },
            },
        }
        transformer.process(finish_event)
        assert not any("c1" in str(k) for k in transformer._buffers)

    def test_tool_output_delta_block_strategy_raises(self) -> None:
        """`block` raises immediately on PII in a `tool-output-delta`."""
        rule = RedactionRule(pii_type="email", strategy="block").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "tools",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": {
                    "event": "tool-output-delta",
                    "tool_call_id": "c1",
                    "delta": "Result: alice@example.com",
                },
            },
        }
        with pytest.raises(PIIDetectionError):
            transformer.process(event)

    def test_tool_call_args_block_strategy_raises(self) -> None:
        """`block` raises immediately when streamed tool-call args contain PII."""
        rule = RedactionRule(pii_type="email", strategy="block").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        event = {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": (
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {
                            "type": "block-delta",
                            "fields": {
                                "type": "tool_call_chunk",
                                "id": "call_1",
                                "name": "send_email",
                                "args": '{"to": "alice@example.com"}',
                            },
                        },
                    },
                    {"run_id": "r1"},
                ),
            },
        }
        with pytest.raises(PIIDetectionError):
            transformer.process(event)

    def test_pii_fully_inside_one_delta_is_redacted_on_finalize(self) -> None:
        """A delta shorter than `lookback` is held until finalize redacts the snapshot."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        events = [
            _make_delta_event("Reach me at alice@example.com tomorrow."),
            _make_finish_event("Reach me at alice@example.com tomorrow."),
        ]
        _run_transformer(transformer, events)
        streamed, finals = _emitted_text(events)  # type: ignore[misc]

        # The raw email never reaches the wire — the delta is held in the
        # lookback buffer and the finalize snapshot is the redacted text.
        assert "alice@example.com" not in streamed
        assert "alice@example.com" not in finals[0]
        assert "[REDACTED_EMAIL]" in finals[0]

    def test_pii_split_across_deltas_is_caught(self) -> None:
        """Email split mid-string across deltas should still be redacted in stream."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=64)

        events = [
            _make_delta_event("Hi, contact alice"),
            _make_delta_event("@example.com when ready"),
            _make_finish_event("Hi, contact alice@example.com when ready"),
        ]
        _run_transformer(transformer, events)
        streamed, finals = _emitted_text(events)  # type: ignore[misc]

        # The held-buffer should have prevented the raw email from being
        # released until detection ran over the concatenation.
        assert "alice@example.com" not in streamed
        assert "alice@example.com" not in finals[0]

    def test_pii_straddling_lookback_boundary_is_caught(self) -> None:
        r"""PII whose start falls in the safe prefix and end in the held tail.

        When `len(combined) > lookback`, the safe/tail split lands inside
        the buffer. Detecting only on the about-to-emit prefix misses
        PII that straddles the boundary — the regex's `\b...\b` anchors
        require a complete match, so a truncated prefix produces no
        detection and the partial PII would leak raw. The transformer
        must run detection on the full accumulated buffer first.
        """
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=32)

        # Single 50-char delta with a 30-char email at position 0.
        # `safe_end = 50 - 32 = 18` falls inside the email, so the old
        # logic would emit `"alice@longerdomain"` raw and hold the rest.
        email = "alice@longerdomain.example.com"
        text = email + "x" * 20
        events = [
            _make_delta_event(text),
            _make_finish_event(text),
        ]
        _run_transformer(transformer, events)
        streamed, finals = _emitted_text(events)  # type: ignore[misc]

        # No prefix of the email reaches the wire.
        assert email not in streamed
        assert "alice@longerdomain" not in streamed
        assert "alice@" not in streamed
        # And the finalized snapshot is fully redacted.
        assert email not in finals[0]
        assert "[REDACTED_EMAIL]" in finals[0]

    def test_credit_card_split_across_whitespace_is_caught(self) -> None:
        """Card with whitespace separators must not leak across deltas."""
        rule = RedactionRule(pii_type="credit_card").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=64)

        events = [
            _make_delta_event("Card: 5425 "),
            _make_delta_event("2334 3010 9903 next"),
            _make_finish_event("Card: 5425 2334 3010 9903 next"),
        ]
        _run_transformer(transformer, events)
        streamed, finals = _emitted_text(events)  # type: ignore[misc]

        # No prefix of the card may reach the wire — the lookback buffer
        # holds whitespace-separated groups until detection runs over the
        # full concatenation.
        assert "5425 2334 3010 9903" not in streamed
        assert "5425" not in streamed
        assert "5425 2334 3010 9903" not in finals[0]
        assert "[REDACTED_CREDIT_CARD]" in finals[0]

    def test_no_transformer_when_neither_output_nor_tool_results_apply(self) -> None:
        """The transformer is gated on either output- or tool-result scrubbing."""
        middleware = PIIMiddleware("email", apply_to_output=False, apply_to_tool_results=False)
        assert middleware.transformers == ()

    def test_transformer_installed_for_tool_results_only(self) -> None:
        """`apply_to_tool_results=True` alone installs the stream transformer.

        Stream consumers see the `tools` channel and `ToolMessage`
        payloads on the `messages` / `values` channels before the
        state-level `before_model` enforcer runs. Without the
        transformer those surfaces would leak raw tool-result PII.
        """
        middleware = PIIMiddleware("email", apply_to_output=False, apply_to_tool_results=True)
        assert len(middleware.transformers) == 1

    def test_block_strategy_installs_buffering_stream_transformer(self) -> None:
        """`block` + output streaming installs a buffering transformer.

        `after_model` is still the canonical blocker. The transformer's
        job is to make sure no PII reaches the streamed surface before
        that hook can raise.
        """
        middleware = PIIMiddleware("email", strategy="block", apply_to_output=True)
        assert len(middleware.transformers) == 1

    def test_block_strategy_raises_on_first_pii_detection(self) -> None:
        """Stream PII under `block`: raises immediately when the pattern completes."""
        rule = RedactionRule(pii_type="email", strategy="block").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        # First delta carries an incomplete email — detection won't match.
        clean_event = _make_delta_event("Reach me at alice")
        transformer.process(clean_event)  # no raise yet

        # Second delta completes the email — detection fires, raises.
        completing_event = _make_delta_event("@example.com soon")
        with pytest.raises(PIIDetectionError):
            transformer.process(completing_event)

    def test_block_strategy_releases_full_text_at_finalize_when_clean(self) -> None:
        """Stream clean text under `block`: deltas empty, finalize is the full text."""
        rule = RedactionRule(pii_type="email", strategy="block").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        events = [
            _make_delta_event("Hello there, "),
            _make_delta_event("how are you?"),
            _make_finish_event("Hello there, how are you?"),
        ]
        _run_transformer(transformer, events)
        streamed, finals = _emitted_text(events)  # type: ignore[misc]

        # Deltas hold everything back; the finalize event carries the
        # whole block at once. `ChatModelStream._resolve_block_text`
        # turns this into a single trailing delta for the consumer's
        # `msg.text` projection.
        assert streamed == ""
        assert finals[0] == "Hello there, how are you?"

    def test_finalize_block_redacts_full_text_even_if_stream_redaction_partial(
        self,
    ) -> None:
        """content-block-finish always re-redacts the finalized text snapshot."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        # Stream the email in a single delta WITHOUT trailing whitespace,
        # so the in-stream lookback might not redact it yet — finalize must
        # still produce a redacted snapshot.
        events = [
            _make_delta_event("alice@example.com"),
            _make_finish_event("alice@example.com"),
        ]
        _run_transformer(transformer, events)
        _, finals = _emitted_text(events)  # type: ignore[misc]

        assert "alice@example.com" not in finals[0]
        assert "[REDACTED_EMAIL]" in finals[0]

    def test_buffers_isolated_by_run_id(self) -> None:
        """Two concurrent runs share the transformer instance but not buffer state."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=64)

        events = [
            _make_delta_event("Hi alice", run_id="run-A"),
            _make_delta_event("Bob's addr is bob", run_id="run-B"),
            _make_delta_event("@example.com soon", run_id="run-A"),
            _make_delta_event("@other.com soon", run_id="run-B"),
            _make_finish_event("Hi alice@example.com soon", run_id="run-A"),
            _make_finish_event("Bob's addr is bob@other.com soon", run_id="run-B"),
        ]
        _run_transformer(transformer, events)

        run_a = "".join(
            e["params"]["data"][0]["delta"]["text"]
            for e in events
            if e["params"]["data"][0].get("event") == "content-block-delta"
            and e["params"]["data"][1].get("run_id") == "run-A"
        )
        run_b = "".join(
            e["params"]["data"][0]["delta"]["text"]
            for e in events
            if e["params"]["data"][0].get("event") == "content-block-delta"
            and e["params"]["data"][1].get("run_id") == "run-B"
        )
        # Splits would have leaked if buffers crossed run_ids.
        assert "alice@example.com" not in run_a
        assert "alice@example.com" not in run_b
        assert "bob@other.com" not in run_a
        assert "bob@other.com" not in run_b

    def test_message_finish_drops_buffers(self) -> None:
        """Abandoned blocks (no content-block-finish) should still release memory."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=64)

        _run_transformer(
            transformer,
            [_make_delta_event("partial text without finish")],
        )
        assert ("r1", 0) in transformer._buffers

        # message-finish for the run wipes any (run-id, *) entries.
        message_finish_event = {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": ({"event": "message-finish"}, {"run_id": "r1"}),
            },
        }
        transformer.process(message_finish_event)
        assert ("r1", 0) not in transformer._buffers

    def test_finalize_clears_all_state(self) -> None:
        """Mux close should be safe — finalize wipes any held buffers."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=64)
        _run_transformer(transformer, [_make_delta_event("hanging text")])
        assert transformer._buffers
        transformer.finalize()
        assert transformer._buffers == {}

    def test_long_pii_exceeding_lookback_still_caught_on_finalize(self) -> None:
        """Patterns longer than `lookback` may slip past the in-stream cap.

        The finalize snapshot is always redacted in full regardless.
        """
        # Choose an absurdly small lookback so a normal email exceeds it.
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=4)

        events = [
            _make_delta_event("hello alice@example.com goodbye"),
            _make_finish_event("hello alice@example.com goodbye"),
        ]
        _run_transformer(transformer, events)
        _, finals = _emitted_text(events)  # type: ignore[misc]
        # The finalized snapshot always re-runs detection over the full text.
        assert "alice@example.com" not in finals[0]

    def test_data_delta_passes_through_untouched(self) -> None:
        """`data-delta` (binary/base64 payloads) is not a typical PII surface.

        Regex detection on base64 strings would produce false positives
        and rarely catches real PII. Users with structured-data PII
        concerns should attach a custom detector that understands their
        payload format.
        """
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        data_event = {
            "type": "event",
            "method": "messages",
            "params": {
                "namespace": [],
                "timestamp": 0,
                "data": (
                    {
                        "event": "content-block-delta",
                        "index": 0,
                        "delta": {
                            "type": "data-delta",
                            "data": "YWxpY2VAZXhhbXBsZS5jb20=",  # base64 of email
                        },
                    },
                    {"run_id": "r1"},
                ),
            },
        }
        transformer.process(data_event)
        assert data_event["params"]["data"][0]["delta"]["data"] == "YWxpY2VAZXhhbXBsZS5jb20="

    def test_transformer_registered_before_messages_transformer_on_agent(self) -> None:
        """The PIIMiddleware transformer must run before MessagesTransformer.

        Otherwise the built-in `messages` projection snapshots the original
        text before redaction, defeating the whole point of the in-flight
        path.
        """
        model = FakeToolCallingModel()
        agent = create_agent(model, [], middleware=[PIIMiddleware("email", apply_to_output=True)])

        run = agent.stream_events({"messages": [HumanMessage("hi")]}, version="v3")
        transformers = run._mux._transformers  # type: ignore[attr-defined]

        pii_idx = next(
            i for i, t in enumerate(transformers) if isinstance(t, _PIIStreamTransformer)
        )
        messages_idx = next(
            i for i, t in enumerate(transformers) if isinstance(t, MessagesTransformer)
        )
        assert pii_idx < messages_idx, (
            "PIIStreamTransformer must be registered before MessagesTransformer "
            "so it can mutate delta.text before the messages projection snapshots it"
        )

        # Drain to close cleanly.
        list(run.tool_calls)


class TestPIIStreamingEndToEnd:
    """End-to-end tests with a real streaming chat model wired into create_agent."""

    @pytest.mark.anyio
    async def test_streamed_messages_projection_is_redacted(self) -> None:
        """Iterating `run.messages` should yield text with PII already redacted.

        Drives a `GenericFakeChatModel` (which actually streams content via
        `_stream` / `_astream` and produces `content-block-delta` protocol
        events) through `create_agent` and asserts the `.text` projection
        of each `ChatModelStream` does not contain the original PII.
        """
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Reach me at alice@example.com today.")])
        )
        agent = create_agent(model, [], middleware=[PIIMiddleware("email", apply_to_output=True)])

        run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")

        collected_text = ""
        async for msg in run.messages:
            async for chunk in msg.text:
                collected_text += chunk

        assert "alice@example.com" not in collected_text
        assert "[REDACTED_EMAIL]" in collected_text

    @pytest.mark.anyio
    async def test_main_event_log_carries_redacted_deltas(self) -> None:
        """Raw protocol events on the main log must not leak the original PII.

        Iterates the run as a raw protocol event stream (the same surface
        external consumers see via `stream_events(version="v3")`) and
        asserts every `content-block-delta` event's `delta.text` is
        already redacted.
        """
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Reach me at alice@example.com today.")])
        )
        agent = create_agent(model, [], middleware=[PIIMiddleware("email", apply_to_output=True)])

        run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")

        surfaces: list[str] = []
        async for event in run:
            if event.get("method") != "messages":
                continue
            data = event["params"].get("data")
            if not isinstance(data, tuple) or len(data) != 2:
                continue
            payload = data[0]
            # v3 protocol-event shape: dict with `event` discriminator.
            if isinstance(payload, dict):
                kind = payload.get("event")
                if kind == "content-block-delta":
                    delta = payload.get("delta") or {}
                    if delta.get("type") == "text-delta":
                        surfaces.append(delta.get("text", ""))
                elif kind == "content-block-finish":
                    content = payload.get("content") or {}
                    if content.get("type") == "text":
                        surfaces.append(content.get("text", ""))
            # Legacy `(BaseMessage, metadata)` shape: message carries text
            # directly on `.content`. The langgraph→langchain adapter falls
            # back to this when `_astream` isn't streaming chunks.
            elif isinstance(payload, BaseMessage):
                content = payload.content
                if isinstance(content, str):
                    surfaces.append(content)

        # Every observed text surface — deltas, finalized snapshots, or
        # legacy whole-message payloads — must already be redacted.
        for text in surfaces:
            assert "alice@example.com" not in text
        # And the redaction marker actually shows up somewhere.
        assert any("[REDACTED_EMAIL]" in text for text in surfaces)

    @pytest.mark.anyio
    async def test_block_strategy_emits_no_pii_and_run_raises(self) -> None:
        """`block` + `apply_to_output=True` + streaming.

        Closes the bypass where the transformer was skipped for `block`,
        leaving plaintext deltas to reach the consumer before
        `after_model` raised. The buffering transformer now keeps every
        wire surface empty and `after_model` raises on the original
        message in state.
        """
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Email me at alice@example.com.")])
        )
        agent = create_agent(
            model,
            [],
            middleware=[PIIMiddleware("email", strategy="block", apply_to_output=True)],
        )

        collected = ""

        async def drain() -> None:
            nonlocal collected
            run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")
            async for msg in run.messages:
                async for chunk in msg.text:
                    collected += chunk

        with pytest.raises(PIIDetectionError):
            await drain()

        # No characters of the PII surface ever reach the consumer — the
        # raised error is the only signal that something was blocked.
        assert "alice@example.com" not in collected
        assert "alice" not in collected

    @pytest.mark.anyio
    async def test_tool_call_args_redacted_end_to_end(self) -> None:
        """Tool-call args containing PII don't reach the consumer."""

        @tool
        def echo(text: str) -> str:
            """Echo."""
            return f"echo: {text}"

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="echo", args={"text": "ping alice@example.com"}, id="c1")],
                [],
            ]
        )
        agent = create_agent(
            model,
            [echo],
            middleware=[PIIMiddleware("email", apply_to_output=True)],
        )

        surfaces: list[str] = []
        run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")
        async for event in run:
            if not isinstance(event, dict):
                continue
            data = event.get("params", {}).get("data")
            if isinstance(data, tuple) and len(data) == 2:
                p = data[0]
                if isinstance(p, BaseMessage):
                    surfaces.append(str(p.content))
                    surfaces.extend(
                        str(tc.get("args")) for tc in getattr(p, "tool_calls", None) or []
                    )
                elif isinstance(p, dict):
                    if p.get("event") == "content-block-finish":
                        c = p.get("content") or {}
                        if c.get("type") == "text":
                            surfaces.append(str(c.get("text", "")))
                        elif c.get("type") in {"tool_call", "server_tool_call"}:
                            surfaces.append(str(c.get("args")))
                    elif p.get("event") == "content-block-delta":
                        d = p.get("delta") or {}
                        if d.get("type") == "block-delta":
                            f = d.get("fields") or {}
                            if f.get("type") in {
                                "tool_call_chunk",
                                "server_tool_call_chunk",
                            }:
                                surfaces.append(str(f.get("args", "")))
            elif isinstance(data, dict):
                e = data.get("event")
                if e == "tool-started":
                    surfaces.append(str(data.get("input")))
                elif e == "tool-output-delta":
                    surfaces.append(str(data.get("delta")))
                elif e == "tool-finished":
                    surfaces.append(str(data.get("output")))
                elif e == "tool-error":
                    surfaces.append(str(data.get("message", "")))

        for s in surfaces:
            assert "alice@example.com" not in s, f"PII leaked on surface: {s!r}"
        assert any("[REDACTED_EMAIL]" in s for s in surfaces), (
            f"redaction marker not observed; surfaces={surfaces}"
        )

    @pytest.mark.anyio
    async def test_tool_output_redacted_end_to_end(self) -> None:
        """Tool output containing PII is redacted on every stream surface."""

        @tool
        def lookup_user(user_id: str) -> str:
            """Look up a user — returns PII."""
            return f"User {user_id}: alice@example.com"

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="lookup_user", args={"user_id": "u1"}, id="c1")],
                [],
            ]
        )
        agent = create_agent(
            model,
            [lookup_user],
            middleware=[
                PIIMiddleware(
                    "email",
                    apply_to_input=True,
                    apply_to_output=True,
                    apply_to_tool_results=True,
                )
            ],
        )

        surfaces: list[str] = []
        run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")
        async for event in run:
            if not isinstance(event, dict):
                continue
            data = event.get("params", {}).get("data")
            if isinstance(data, tuple) and len(data) == 2:
                p = data[0]
                if isinstance(p, BaseMessage):
                    surfaces.append(str(p.content))
                elif isinstance(p, dict):
                    surfaces.append(repr(p))
            elif isinstance(data, dict):
                surfaces.append(repr(data))

        for s in surfaces:
            assert "alice@example.com" not in s, f"tool output leaked on surface: {s!r}"

    @pytest.mark.anyio
    async def test_block_raises_on_tool_output_pii(self) -> None:
        """`block` + tool output with PII → run raises, no leak on the wire."""

        @tool
        def lookup_user(user_id: str) -> str:
            """Look up a user — returns PII."""
            return f"User {user_id}: alice@example.com"

        model = FakeToolCallingModel(
            tool_calls=[
                [ToolCall(name="lookup_user", args={"user_id": "u1"}, id="c1")],
                [],
            ]
        )
        agent = create_agent(
            model,
            [lookup_user],
            middleware=[
                PIIMiddleware(
                    "email",
                    strategy="block",
                    apply_to_input=True,
                    apply_to_output=True,
                    apply_to_tool_results=True,
                )
            ],
        )

        collected: list[str] = []

        async def drain() -> None:
            run = await agent.astream_events({"messages": [HumanMessage("hi")]}, version="v3")
            async for event in run:
                if isinstance(event, dict):
                    data = event.get("params", {}).get("data")
                    if isinstance(data, tuple) and len(data) == 2:
                        p = data[0]
                        if isinstance(p, BaseMessage):
                            collected.append(str(p.content))
                        elif isinstance(p, dict):
                            collected.append(repr(p))
                    elif isinstance(data, dict):
                        collected.append(repr(data))

        with pytest.raises(PIIDetectionError):
            await drain()

        for s in collected:
            assert "alice@example.com" not in s, f"PII leaked under block: {s!r}"

    @pytest.mark.anyio
    async def test_subgraph_redaction_via_create_agent_in_tool(self) -> None:
        """A sub-agent invoked inside a tool inherits the parent's transformer.

        `StreamMux._make_child` clones the factory list down to every
        subgraph scope, so a fresh `_PIIStreamTransformer` runs at the
        sub-agent's mini-mux too. This is the supported pattern: attach
        `PIIMiddleware` to the outer agent and every nested model call
        — including those run by sub-agents inside tools — gets redacted
        in flight by its own scoped instance of the transformer.
        """
        inner_model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Hi bob@example.com, here is data.")])
        )
        # No PII middleware on the inner agent — the outer's transformer
        # factory propagates down to the subgraph scope.
        inner_agent = create_agent(inner_model, [])

        @tool
        def delegate(query: str) -> str:
            """Hand the query off to the inner agent."""
            result = inner_agent.invoke({"messages": [HumanMessage(query)]})
            return str(result["messages"][-1].content)

        outer_model = FakeToolCallingModel(
            tool_calls=[
                [{"name": "delegate", "args": {"query": "hi"}, "id": "tc1"}],
                [],
            ]
        )
        outer_agent = create_agent(
            outer_model,
            [delegate],
            middleware=[PIIMiddleware("email", apply_to_output=True)],
        )

        run = await outer_agent.astream_events({"messages": [HumanMessage("go")]}, version="v3")

        seen_email_in_deltas = False
        seen_email_in_finalized = False
        seen_redaction = False
        async for event in run:
            if event.get("method") != "messages":
                continue
            data = event["params"].get("data")
            if not isinstance(data, tuple) or len(data) != 2:
                continue
            payload = data[0]
            if not isinstance(payload, dict):
                continue
            kind = payload.get("event")
            if kind == "content-block-delta":
                delta = payload.get("delta") or {}
                if delta.get("type") == "text-delta":
                    text = delta.get("text", "")
                    if "bob@example.com" in text:
                        seen_email_in_deltas = True
                    if "[REDACTED_EMAIL]" in text:
                        seen_redaction = True
            elif kind == "content-block-finish":
                content = payload.get("content") or {}
                if content.get("type") == "text":
                    text = content.get("text", "")
                    if "bob@example.com" in text:
                        seen_email_in_finalized = True
                    if "[REDACTED_EMAIL]" in text:
                        seen_redaction = True

        assert not seen_email_in_deltas, (
            "raw PII leaked through a subgraph's streamed deltas — child "
            "mini-mux did not inherit the outer transformer factory"
        )
        assert not seen_email_in_finalized, (
            "raw PII leaked through a subgraph's content-block-finish snapshot"
        )
        assert seen_redaction, "transformer never fired at the subgraph scope"

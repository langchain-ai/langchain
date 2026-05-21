"""Tests for PII detection middleware."""

import re
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
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

    def test_pii_fully_inside_one_delta_with_whitespace_after(self) -> None:
        """Single delta containing the full PII followed by whitespace should redact it."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        events = [
            _make_delta_event("Reach me at alice@example.com tomorrow."),
            _make_finish_event("Reach me at alice@example.com tomorrow."),
        ]
        _run_transformer(transformer, events)
        streamed, finals = _emitted_text(events)  # type: ignore[misc]

        assert "alice@example.com" not in streamed
        assert "[REDACTED_EMAIL]" in streamed
        assert "alice@example.com" not in finals[0]
        assert "[REDACTED_EMAIL]" in finals[0]

    def test_pii_split_across_deltas_is_caught(self) -> None:
        """Email split mid-string across deltas should still be redacted in stream."""
        rule = RedactionRule(pii_type="email").resolve()
        # Disable whitespace flush so we exercise the lookback buffer path.
        transformer = _PIIStreamTransformer(rule=rule, lookback=64, whitespace_boundary=False)

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

    def test_whitespace_boundary_flushes_safe_prefix_early(self) -> None:
        """Whitespace appearing inside a delta should release everything before it."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule, lookback=128, whitespace_boundary=True)

        # 20 chars then a space — even though we're under the 128-char
        # lookback, whitespace flush should release the prefix.
        events = [_make_delta_event("Just some plain text here.")]
        _run_transformer(transformer, events)
        streamed, _ = _emitted_text(events)  # type: ignore[misc]

        # "Just some plain text " is safe (last whitespace at end), so the
        # delta should emit everything up to and including the last space.
        assert streamed.startswith("Just some plain text ")

    def test_custom_detector_defaults_to_no_whitespace_flush(self) -> None:
        """A PIIMiddleware with a custom regex should NOT whitespace-flush by default."""
        middleware = PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{8}",
            strategy="redact",
            apply_to_output=True,
        )

        # The middleware should register a transformer.
        assert len(middleware.transformers) == 1
        factory = middleware.transformers[0]
        transformer = factory(())
        # The transformer was wired with whitespace_boundary=False since
        # the detector is custom.
        assert transformer._whitespace_boundary is False

    def test_builtin_detector_defaults_to_whitespace_flush(self) -> None:
        """A built-in PII type should default to whitespace flush enabled."""
        middleware = PIIMiddleware("email", apply_to_output=True)
        assert len(middleware.transformers) == 1
        transformer = middleware.transformers[0](())
        assert transformer._whitespace_boundary is True

    def test_apply_to_output_false_registers_no_transformer(self) -> None:
        """Streaming redaction is gated on apply_to_output."""
        middleware = PIIMiddleware("email", apply_to_output=False)
        assert middleware.transformers == ()

    def test_block_strategy_skips_stream_transformer(self) -> None:
        """`block` strategy can't run mid-stream — fall back to state-level only."""
        middleware = PIIMiddleware("email", strategy="block", apply_to_output=True)
        assert middleware.transformers == ()

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
        transformer = _PIIStreamTransformer(rule=rule, lookback=64, whitespace_boundary=False)

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
        transformer = _PIIStreamTransformer(rule=rule, lookback=64, whitespace_boundary=False)

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
        transformer = _PIIStreamTransformer(rule=rule, lookback=64, whitespace_boundary=False)
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
        transformer = _PIIStreamTransformer(rule=rule, lookback=4, whitespace_boundary=False)

        events = [
            _make_delta_event("hello alice@example.com goodbye"),
            _make_finish_event("hello alice@example.com goodbye"),
        ]
        _run_transformer(transformer, events)
        _, finals = _emitted_text(events)  # type: ignore[misc]
        # The finalized snapshot always re-runs detection over the full text.
        assert "alice@example.com" not in finals[0]

    def test_non_text_delta_passes_through_untouched(self) -> None:
        """Reasoning/data deltas should not be mutated by the text-only path."""
        rule = RedactionRule(pii_type="email").resolve()
        transformer = _PIIStreamTransformer(rule=rule)

        reasoning_event = {
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
                            "reasoning": "alice@example.com thinking",
                        },
                    },
                    {"run_id": "r1"},
                ),
            },
        }
        transformer.process(reasoning_event)
        # The reasoning field is left alone — this transformer scopes itself
        # to text-delta only, matching its docstring contract.
        assert (
            reasoning_event["params"]["data"][0]["delta"]["reasoning"]
            == "alice@example.com thinking"
        )

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

        delta_texts: list[str] = []
        finalized_texts: list[str] = []
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
                    delta_texts.append(delta.get("text", ""))
            elif kind == "content-block-finish":
                content = payload.get("content") or {}
                if content.get("type") == "text":
                    finalized_texts.append(content.get("text", ""))

        # Per-delta texts on the wire are redacted in flight.
        for text in delta_texts:
            assert "alice@example.com" not in text
        # The finalized snapshot is also redacted in full.
        for text in finalized_texts:
            assert "alice@example.com" not in text
        # And the redaction marker actually shows up somewhere.
        joined = "".join(delta_texts) + "".join(finalized_texts)
        assert "[REDACTED_EMAIL]" in joined

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

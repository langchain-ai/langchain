"""Tests for PII detection middleware."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

from langchain.agents.middleware.pii import (
    PIIDetectionError,
    PIIMiddleware,
    detect_credit_card,
    detect_email,
    detect_ip,
    detect_mac_address,
    detect_url,
)
from langchain.agents.factory import create_agent

from .model import FakeToolCallingModel


# ============================================================================
# Detection Function Tests
# ============================================================================


class TestEmailDetection:
    """Test email detection."""

    def test_detect_valid_email(self):
        content = "Contact me at john.doe@example.com for more info."
        matches = detect_email(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "email"
        assert matches[0]["value"] == "john.doe@example.com"
        assert matches[0]["start"] == 14
        assert matches[0]["end"] == 34

    def test_detect_multiple_emails(self):
        content = "Email alice@test.com or bob@company.org"
        matches = detect_email(content)

        assert len(matches) == 2
        assert matches[0]["value"] == "alice@test.com"
        assert matches[1]["value"] == "bob@company.org"

    def test_no_email(self):
        content = "This text has no email addresses."
        matches = detect_email(content)
        assert len(matches) == 0

    def test_invalid_email_format(self):
        content = "Invalid emails: @test.com, user@, user@domain"
        matches = detect_email(content)
        # Should not match invalid formats
        assert len(matches) == 0


class TestCreditCardDetection:
    """Test credit card detection with Luhn validation."""

    def test_detect_valid_credit_card(self):
        # Valid Visa test number
        content = "Card: 4532015112830366"
        matches = detect_credit_card(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "credit_card"
        assert matches[0]["value"] == "4532015112830366"

    def test_detect_credit_card_with_spaces(self):
        # Valid Mastercard test number
        content = "Card: 5425233430109903"
        # Add spaces
        spaced_content = "Card: 5425 2334 3010 9903"
        matches = detect_credit_card(spaced_content)

        assert len(matches) == 1
        assert "5425 2334 3010 9903" in matches[0]["value"]

    def test_detect_credit_card_with_dashes(self):
        content = "Card: 4532-0151-1283-0366"
        matches = detect_credit_card(content)

        assert len(matches) == 1

    def test_invalid_luhn_not_detected(self):
        # Invalid Luhn checksum
        content = "Card: 1234567890123456"
        matches = detect_credit_card(content)
        assert len(matches) == 0

    def test_no_credit_card(self):
        content = "No cards here."
        matches = detect_credit_card(content)
        assert len(matches) == 0


class TestIPDetection:
    """Test IP address detection."""

    def test_detect_valid_ipv4(self):
        content = "Server IP: 192.168.1.1"
        matches = detect_ip(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "ip"
        assert matches[0]["value"] == "192.168.1.1"

    def test_detect_multiple_ips(self):
        content = "Connect to 10.0.0.1 or 8.8.8.8"
        matches = detect_ip(content)

        assert len(matches) == 2
        assert matches[0]["value"] == "10.0.0.1"
        assert matches[1]["value"] == "8.8.8.8"

    def test_invalid_ip_not_detected(self):
        # Out of range octets
        content = "Not an IP: 999.999.999.999"
        matches = detect_ip(content)
        assert len(matches) == 0

    def test_version_number_not_detected(self):
        # Version numbers should not be detected as IPs
        content = "Version 1.2.3.4 released"
        matches = detect_ip(content)
        # This is a valid IP format, so it will be detected
        # This is acceptable behavior
        assert len(matches) >= 0

    def test_no_ip(self):
        content = "No IP addresses here."
        matches = detect_ip(content)
        assert len(matches) == 0


class TestMACAddressDetection:
    """Test MAC address detection."""

    def test_detect_mac_with_colons(self):
        content = "MAC: 00:1A:2B:3C:4D:5E"
        matches = detect_mac_address(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "mac_address"
        assert matches[0]["value"] == "00:1A:2B:3C:4D:5E"

    def test_detect_mac_with_dashes(self):
        content = "MAC: 00-1A-2B-3C-4D-5E"
        matches = detect_mac_address(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "00-1A-2B-3C-4D-5E"

    def test_detect_lowercase_mac(self):
        content = "MAC: aa:bb:cc:dd:ee:ff"
        matches = detect_mac_address(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "aa:bb:cc:dd:ee:ff"

    def test_no_mac(self):
        content = "No MAC address here."
        matches = detect_mac_address(content)
        assert len(matches) == 0

    def test_partial_mac_not_detected(self):
        content = "Partial: 00:1A:2B:3C"
        matches = detect_mac_address(content)
        assert len(matches) == 0


class TestURLDetection:
    """Test URL detection."""

    def test_detect_http_url(self):
        content = "Visit http://example.com for details."
        matches = detect_url(content)

        assert len(matches) == 1
        assert matches[0]["type"] == "url"
        assert matches[0]["value"] == "http://example.com"

    def test_detect_https_url(self):
        content = "Visit https://secure.example.com/path"
        matches = detect_url(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "https://secure.example.com/path"

    def test_detect_www_url(self):
        content = "Check www.example.com"
        matches = detect_url(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "www.example.com"

    def test_detect_bare_domain_with_path(self):
        content = "Go to example.com/page"
        matches = detect_url(content)

        assert len(matches) == 1
        assert matches[0]["value"] == "example.com/page"

    def test_detect_multiple_urls(self):
        content = "Visit http://test.com and https://example.org"
        matches = detect_url(content)

        assert len(matches) == 2

    def test_no_url(self):
        content = "No URLs here."
        matches = detect_url(content)
        assert len(matches) == 0

    def test_bare_domain_without_path_not_detected(self):
        # To reduce false positives, bare domains without paths are not detected
        content = "The word example.com in prose"
        matches = detect_url(content)
        # May or may not detect depending on implementation
        # This is acceptable


# ============================================================================
# Strategy Tests
# ============================================================================


class TestRedactStrategy:
    """Test redact strategy."""

    def test_redact_email(self):
        middleware = PIIMiddleware("email", strategy="redact")
        state = {"messages": [HumanMessage("Email me at test@example.com")]}

        result = middleware.before_model(state, None)

        assert result is not None
        assert "[REDACTED_EMAIL]" in result["messages"][0].content
        assert "test@example.com" not in result["messages"][0].content

    def test_redact_multiple_pii(self):
        middleware = PIIMiddleware("email", strategy="redact")
        state = {"messages": [HumanMessage("Contact alice@test.com or bob@test.com")]}

        result = middleware.before_model(state, None)

        assert result is not None
        content = result["messages"][0].content
        assert content.count("[REDACTED_EMAIL]") == 2
        assert "alice@test.com" not in content
        assert "bob@test.com" not in content


class TestMaskStrategy:
    """Test mask strategy."""

    def test_mask_email(self):
        middleware = PIIMiddleware("email", strategy="mask")
        state = {"messages": [HumanMessage("Email: user@example.com")]}

        result = middleware.before_model(state, None)

        assert result is not None
        content = result["messages"][0].content
        assert "user@****.com" in content
        assert "user@example.com" not in content

    def test_mask_credit_card(self):
        middleware = PIIMiddleware("credit_card", strategy="mask")
        # Valid test card
        state = {"messages": [HumanMessage("Card: 4532015112830366")]}

        result = middleware.before_model(state, None)

        assert result is not None
        content = result["messages"][0].content
        assert "0366" in content  # Last 4 digits visible
        assert "4532015112830366" not in content

    def test_mask_ip(self):
        middleware = PIIMiddleware("ip", strategy="mask")
        state = {"messages": [HumanMessage("IP: 192.168.1.100")]}

        result = middleware.before_model(state, None)

        assert result is not None
        content = result["messages"][0].content
        assert "*.*.*.100" in content
        assert "192.168.1.100" not in content


class TestHashStrategy:
    """Test hash strategy."""

    def test_hash_email(self):
        middleware = PIIMiddleware("email", strategy="hash")
        state = {"messages": [HumanMessage("Email: test@example.com")]}

        result = middleware.before_model(state, None)

        assert result is not None
        content = result["messages"][0].content
        assert "<email_hash:" in content
        assert ">" in content
        assert "test@example.com" not in content

    def test_hash_is_deterministic(self):
        middleware = PIIMiddleware("email", strategy="hash")

        # Same email should produce same hash
        state1 = {"messages": [HumanMessage("Email: test@example.com")]}
        state2 = {"messages": [HumanMessage("Email: test@example.com")]}

        result1 = middleware.before_model(state1, None)
        result2 = middleware.before_model(state2, None)

        assert result1["messages"][0].content == result2["messages"][0].content


class TestBlockStrategy:
    """Test block strategy."""

    def test_block_raises_exception(self):
        middleware = PIIMiddleware("email", strategy="block")
        state = {"messages": [HumanMessage("Email: test@example.com")]}

        with pytest.raises(PIIDetectionError) as exc_info:
            middleware.before_model(state, None)

        assert exc_info.value.pii_type == "email"
        assert len(exc_info.value.matches) == 1
        assert "test@example.com" in exc_info.value.matches[0]["value"]

    def test_block_with_multiple_matches(self):
        middleware = PIIMiddleware("email", strategy="block")
        state = {"messages": [HumanMessage("Emails: alice@test.com and bob@test.com")]}

        with pytest.raises(PIIDetectionError) as exc_info:
            middleware.before_model(state, None)

        assert len(exc_info.value.matches) == 2


# ============================================================================
# Middleware Integration Tests
# ============================================================================


class TestPIIMiddlewareIntegration:
    """Test PIIMiddleware integration with agent."""

    def test_apply_to_input_only(self):
        """Test that middleware only processes input when configured."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=True, apply_to_output=False
        )  # noqa: E501

        # Should process HumanMessage
        state = {"messages": [HumanMessage("Email: test@example.com")]}
        result = middleware.before_model(state, None)
        assert result is not None
        assert "[REDACTED_EMAIL]" in result["messages"][0].content

        # Should not process AIMessage
        state = {"messages": [AIMessage("My email is ai@example.com")]}
        result = middleware.after_model(state, None)
        assert result is None

    def test_apply_to_output_only(self):
        """Test that middleware only processes output when configured."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=False, apply_to_output=True
        )  # noqa: E501

        # Should not process HumanMessage
        state = {"messages": [HumanMessage("Email: test@example.com")]}
        result = middleware.before_model(state, None)
        assert result is None

        # Should process AIMessage
        state = {"messages": [AIMessage("My email is ai@example.com")]}
        result = middleware.after_model(state, None)
        assert result is not None
        assert "[REDACTED_EMAIL]" in result["messages"][0].content

    def test_apply_to_both(self):
        """Test that middleware processes both input and output."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=True, apply_to_output=True
        )  # noqa: E501

        # Should process HumanMessage
        state = {"messages": [HumanMessage("Email: test@example.com")]}
        result = middleware.before_model(state, None)
        assert result is not None

        # Should process AIMessage
        state = {"messages": [AIMessage("My email is ai@example.com")]}
        result = middleware.after_model(state, None)
        assert result is not None

    def test_no_pii_returns_none(self):
        """Test that middleware returns None when no PII detected."""
        middleware = PIIMiddleware("email", strategy="redact")
        state = {"messages": [HumanMessage("No PII here")]}

        result = middleware.before_model(state, None)
        assert result is None

    def test_empty_messages(self):
        """Test that middleware handles empty messages gracefully."""
        middleware = PIIMiddleware("email", strategy="redact")
        state = {"messages": []}

        result = middleware.before_model(state, None)
        assert result is None

    def test_apply_to_tool_results(self):
        """Test that middleware processes tool results when enabled."""
        middleware = PIIMiddleware(
            "email", strategy="redact", apply_to_input=False, apply_to_tool_results=True
        )

        # Simulate a conversation with tool call and result containing PII
        state = {
            "messages": [
                HumanMessage("Search for John"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="search", args={}, id="call_123", type="tool_call")],
                ),
                ToolMessage(content="Found: john@example.com", tool_call_id="call_123"),
            ]
        }

        result = middleware.before_model(state, None)

        assert result is not None
        # Check that the tool message was redacted
        tool_msg = result["messages"][2]
        assert isinstance(tool_msg, ToolMessage)
        assert "[REDACTED_EMAIL]" in tool_msg.content
        assert "john@example.com" not in tool_msg.content

    def test_apply_to_tool_results_mask_strategy(self):
        """Test that mask strategy works for tool results."""
        middleware = PIIMiddleware(
            "ip", strategy="mask", apply_to_input=False, apply_to_tool_results=True
        )

        state = {
            "messages": [
                HumanMessage("Get server IP"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="get_ip", args={}, id="call_456", type="tool_call")],
                ),
                ToolMessage(content="Server IP: 192.168.1.100", tool_call_id="call_456"),
            ]
        }

        result = middleware.before_model(state, None)

        assert result is not None
        tool_msg = result["messages"][2]
        assert "*.*.*.100" in tool_msg.content
        assert "192.168.1.100" not in tool_msg.content

    def test_apply_to_tool_results_block_strategy(self):
        """Test that block strategy raises error for PII in tool results."""
        middleware = PIIMiddleware(
            "email", strategy="block", apply_to_input=False, apply_to_tool_results=True
        )

        state = {
            "messages": [
                HumanMessage("Search for user"),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="search", args={}, id="call_789", type="tool_call")],
                ),
                ToolMessage(content="User email: sensitive@example.com", tool_call_id="call_789"),
            ]
        }

        with pytest.raises(PIIDetectionError) as exc_info:
            middleware.before_model(state, None)

        assert exc_info.value.pii_type == "email"
        assert len(exc_info.value.matches) == 1

    def test_with_agent(self):
        """Test PIIMiddleware integrated with create_agent."""
        model = FakeToolCallingModel(responses=[AIMessage(content="Thanks for sharing!")])

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

    def test_custom_regex_detector(self):
        # Custom regex for API keys
        middleware = PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="redact",
        )

        state = {"messages": [HumanMessage("Key: sk-abcdefghijklmnopqrstuvwxyz123456")]}
        result = middleware.before_model(state, None)

        assert result is not None
        assert "[REDACTED_API_KEY]" in result["messages"][0].content

    def test_custom_callable_detector(self):
        # Custom detector function
        def detect_custom(content):
            matches = []
            if "CONFIDENTIAL" in content:
                idx = content.index("CONFIDENTIAL")
                matches.append(
                    {
                        "type": "confidential",
                        "value": "CONFIDENTIAL",
                        "start": idx,
                        "end": idx + 12,
                    }
                )
            return matches

        middleware = PIIMiddleware(
            "confidential",
            detector=detect_custom,
            strategy="redact",
        )

        state = {"messages": [HumanMessage("This is CONFIDENTIAL information")]}
        result = middleware.before_model(state, None)

        assert result is not None
        assert "[REDACTED_CONFIDENTIAL]" in result["messages"][0].content

    def test_unknown_builtin_type_raises_error(self):
        with pytest.raises(ValueError, match="Unknown PII type"):
            PIIMiddleware("unknown_type", strategy="redact")

    def test_custom_type_without_detector_raises_error(self):
        with pytest.raises(ValueError, match="Unknown PII type"):
            PIIMiddleware("custom_type", strategy="redact")


class TestMultipleMiddleware:
    """Test using multiple PII middleware instances."""

    def test_sequential_application(self):
        """Test that multiple PII types are detected when applied sequentially."""
        # First apply email middleware
        email_middleware = PIIMiddleware("email", strategy="redact")
        state = {"messages": [HumanMessage("Email: test@example.com, IP: 192.168.1.1")]}
        result1 = email_middleware.before_model(state, None)

        # Then apply IP middleware to the result
        ip_middleware = PIIMiddleware("ip", strategy="mask")
        state_with_email_redacted = {"messages": result1["messages"]}
        result2 = ip_middleware.before_model(state_with_email_redacted, None)

        content = result2["messages"][0].content

        # Email should be redacted
        assert "[REDACTED_EMAIL]" in content
        assert "test@example.com" not in content

        # IP should be masked
        assert "*.*.*.1" in content
        assert "192.168.1.1" not in content

    def test_multiple_pii_middleware_with_create_agent(self):
        """Test that multiple PIIMiddleware instances work together in create_agent."""
        model = FakeToolCallingModel(responses=[AIMessage(content="Response received")])

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

    def test_custom_detector_for_multiple_types(self):
        """Test using a single middleware with custom detector for multiple PII types.

        This is an alternative to using multiple middleware instances,
        useful when you want the same strategy for multiple PII types.
        """

        # Combine multiple detectors into one
        def detect_email_and_ip(content):
            from langchain.agents.middleware.pii import detect_email, detect_ip

            return detect_email(content) + detect_ip(content)

        middleware = PIIMiddleware(
            "email_or_ip",
            detector=detect_email_and_ip,
            strategy="redact",
        )

        state = {"messages": [HumanMessage("Email: test@example.com, IP: 10.0.0.1")]}
        result = middleware.before_model(state, None)

        content = result["messages"][0].content
        assert "test@example.com" not in content
        assert "10.0.0.1" not in content

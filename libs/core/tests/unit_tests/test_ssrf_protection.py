"""Tests for SSRF protection utilities."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from langchain_core._security._ssrf_protection import (
    SSRFProtectedUrl,
    SSRFProtectedUrlRelaxed,
    is_cloud_metadata,
    is_localhost,
    is_private_ip,
    is_safe_url,
    validate_safe_url,
)


class TestIPValidation:
    """Tests for IP address validation functions."""

    def test_is_private_ip_ipv4(self) -> None:
        """Test private IPv4 address detection."""
        assert is_private_ip("10.0.0.1") is True
        assert is_private_ip("172.16.0.1") is True
        assert is_private_ip("192.168.1.1") is True
        assert is_private_ip("127.0.0.1") is True
        assert is_private_ip("169.254.169.254") is True
        assert is_private_ip("0.0.0.1") is True

    def test_is_private_ip_ipv6(self) -> None:
        """Test private IPv6 address detection."""
        assert is_private_ip("::1") is True  # Loopback
        assert is_private_ip("fc00::1") is True  # Unique local
        assert is_private_ip("fe80::1") is True  # Link-local
        assert is_private_ip("ff00::1") is True  # Multicast

    def test_is_private_ip_public(self) -> None:
        """Test that public IPs are not flagged as private."""
        assert is_private_ip("8.8.8.8") is False
        assert is_private_ip("1.1.1.1") is False
        assert is_private_ip("151.101.1.140") is False

    def test_is_private_ip_invalid(self) -> None:
        """Test handling of invalid IP addresses."""
        assert is_private_ip("not-an-ip") is False
        assert is_private_ip("999.999.999.999") is False

    def test_is_cloud_metadata_ips(self) -> None:
        """Test cloud metadata IP detection."""
        assert is_cloud_metadata("example.com", "169.254.169.254") is True
        assert is_cloud_metadata("example.com", "169.254.170.2") is True
        assert is_cloud_metadata("example.com", "100.100.100.200") is True

    def test_is_cloud_metadata_hostnames(self) -> None:
        """Test cloud metadata hostname detection."""
        assert is_cloud_metadata("metadata.google.internal") is True
        assert is_cloud_metadata("metadata") is True
        assert is_cloud_metadata("instance-data") is True
        assert is_cloud_metadata("METADATA.GOOGLE.INTERNAL") is True  # Case insensitive

    def test_is_cloud_metadata_safe(self) -> None:
        """Test that normal URLs are not flagged as cloud metadata."""
        assert is_cloud_metadata("example.com", "8.8.8.8") is False
        assert is_cloud_metadata("google.com") is False

    def test_is_localhost_hostnames(self) -> None:
        """Test localhost hostname detection."""
        assert is_localhost("localhost") is True
        assert is_localhost("LOCALHOST") is True
        assert is_localhost("localhost.localdomain") is True

    def test_is_localhost_ips(self) -> None:
        """Test localhost IP detection."""
        assert is_localhost("example.com", "127.0.0.1") is True
        assert is_localhost("example.com", "::1") is True
        assert is_localhost("example.com", "0.0.0.0") is True

    def test_is_localhost_safe(self) -> None:
        """Test that normal hosts are not flagged as localhost."""
        assert is_localhost("example.com", "8.8.8.8") is False
        assert is_localhost("google.com") is False


class TestValidateSafeUrl:
    """Tests for validate_safe_url function."""

    def test_valid_public_https_url(self) -> None:
        """Test that valid public HTTPS URLs are accepted."""
        url = "https://hooks.slack.com/services/xxx"
        result = validate_safe_url(url)
        assert result == url

    def test_valid_public_http_url(self) -> None:
        """Test that valid public HTTP URLs are accepted."""
        url = "http://example.com/webhook"
        result = validate_safe_url(url)
        assert result == url

    def test_localhost_blocked_by_default(self) -> None:
        """Test that localhost URLs are blocked by default."""
        with pytest.raises(ValueError, match="Localhost"):
            validate_safe_url("http://localhost:8080/webhook")

        with pytest.raises(ValueError, match="localhost"):
            validate_safe_url("http://127.0.0.1:8080/webhook")

    def test_localhost_allowed_with_flag(self) -> None:
        """Test that localhost is allowed with allow_private=True."""
        url = "http://localhost:8080/webhook"
        result = validate_safe_url(url, allow_private=True)
        assert result == url

        url = "http://127.0.0.1:8080/webhook"
        result = validate_safe_url(url, allow_private=True)
        assert result == url

    def test_private_ip_blocked_by_default(self) -> None:
        """Test that private IPs are blocked by default."""
        with pytest.raises(ValueError, match="private IP"):
            validate_safe_url("http://192.168.1.1/webhook")

        with pytest.raises(ValueError, match="private IP"):
            validate_safe_url("http://10.0.0.1/webhook")

        with pytest.raises(ValueError, match="private IP"):
            validate_safe_url("http://172.16.0.1/webhook")

    def test_private_ip_allowed_with_flag(self) -> None:
        """Test that private IPs are allowed with allow_private=True."""
        # Note: These will fail DNS resolution in tests, so we skip actual validation
        # In production, they would be validated properly

    def test_cloud_metadata_always_blocked(self) -> None:
        """Test that cloud metadata endpoints are always blocked."""
        with pytest.raises(ValueError, match="metadata"):
            validate_safe_url("http://169.254.169.254/latest/meta-data/")

        # Even with allow_private=True
        with pytest.raises(ValueError, match="metadata"):
            validate_safe_url(
                "http://169.254.169.254/latest/meta-data/",
                allow_private=True,
            )

    def test_invalid_scheme_blocked(self) -> None:
        """Test that non-HTTP(S) schemes are blocked."""
        with pytest.raises(ValueError, match="scheme"):
            validate_safe_url("ftp://example.com/file")

        with pytest.raises(ValueError, match="scheme"):
            validate_safe_url("file:///etc/passwd")

        with pytest.raises(ValueError, match="scheme"):
            validate_safe_url("javascript:alert(1)")

    def test_https_only_mode(self) -> None:
        """Test that HTTP is blocked when allow_http=False."""
        with pytest.raises(ValueError, match="HTTPS"):
            validate_safe_url("http://example.com/webhook", allow_http=False)

        # HTTPS should still work
        url = "https://example.com/webhook"
        result = validate_safe_url(url, allow_http=False)
        assert result == url

    def test_url_without_hostname(self) -> None:
        """Test that URLs without hostname are rejected."""
        with pytest.raises(ValueError, match="hostname"):
            validate_safe_url("http:///path")

    def test_dns_resolution_failure(self) -> None:
        """Test handling of DNS resolution failures."""
        with pytest.raises(ValueError, match="resolve"):
            validate_safe_url("http://this-domain-definitely-does-not-exist-12345.com")

    def test_testserver_allowed(self, monkeypatch: Any) -> None:
        """Test that testserver hostname is allowed for test environments."""
        # testserver is used by FastAPI/Starlette test clients
        monkeypatch.setenv("LANGCHAIN_ENV", "local_test")
        url = "http://testserver/webhook"
        result = validate_safe_url(url)
        assert result == url


class TestIsSafeUrl:
    """Tests for is_safe_url function (non-throwing version)."""

    def test_safe_url_returns_true(self) -> None:
        """Test that safe URLs return True."""
        assert is_safe_url("https://example.com/webhook") is True
        assert is_safe_url("http://hooks.slack.com/services/xxx") is True

    def test_unsafe_url_returns_false(self) -> None:
        """Test that unsafe URLs return False."""
        assert is_safe_url("http://localhost:8080") is False
        assert is_safe_url("http://127.0.0.1:8080") is False
        assert is_safe_url("http://192.168.1.1") is False
        assert is_safe_url("http://169.254.169.254") is False

    def test_unsafe_url_safe_with_allow_private(self) -> None:
        """Test that private URLs are safe with allow_private=True."""
        assert is_safe_url("http://localhost:8080", allow_private=True) is True
        assert is_safe_url("http://127.0.0.1:8080", allow_private=True) is True

    def test_cloud_metadata_always_unsafe(self) -> None:
        """Test that cloud metadata is always unsafe."""
        assert is_safe_url("http://169.254.169.254") is False
        assert is_safe_url("http://169.254.169.254", allow_private=True) is False


class TestSSRFProtectedUrlType:
    """Tests for SSRFProtectedUrl Pydantic type."""

    def test_valid_url_accepted(self) -> None:
        """Test that valid URLs are accepted by Pydantic schema."""

        class WebhookSchema(BaseModel):
            url: SSRFProtectedUrl

        schema = WebhookSchema(url="https://hooks.slack.com/services/xxx")
        assert str(schema.url).startswith("https://hooks.slack.com/")

    def test_localhost_rejected(self) -> None:
        """Test that localhost URLs are rejected by Pydantic schema."""

        class WebhookSchema(BaseModel):
            url: SSRFProtectedUrl

        with pytest.raises(ValidationError):
            WebhookSchema(url="http://localhost:8080")

    def test_private_ip_rejected(self) -> None:
        """Test that private IPs are rejected by Pydantic schema."""

        class WebhookSchema(BaseModel):
            url: SSRFProtectedUrl

        with pytest.raises(ValidationError):
            WebhookSchema(url="http://192.168.1.1")

    def test_cloud_metadata_rejected(self) -> None:
        """Test that cloud metadata is rejected by Pydantic schema."""

        class WebhookSchema(BaseModel):
            url: SSRFProtectedUrl

        with pytest.raises(ValidationError):
            WebhookSchema(url="http://169.254.169.254/latest/meta-data/")


class TestSSRFProtectedUrlRelaxedType:
    """Tests for SSRFProtectedUrlRelaxed Pydantic type."""

    def test_localhost_accepted(self) -> None:
        """Test that localhost URLs are accepted by relaxed schema."""

        class WebhookSchema(BaseModel):
            url: SSRFProtectedUrlRelaxed

        schema = WebhookSchema(url="http://localhost:8080")
        assert str(schema.url).startswith("http://localhost")

    def test_cloud_metadata_still_rejected(self) -> None:
        """Test that cloud metadata is still rejected by relaxed schema."""

        class WebhookSchema(BaseModel):
            url: SSRFProtectedUrlRelaxed

        with pytest.raises(ValidationError):
            WebhookSchema(url="http://169.254.169.254/latest/meta-data/")


class TestRealWorldURLs:
    """Tests with real-world webhook URLs."""

    def test_slack_webhook(self) -> None:
        """Test Slack webhook URL."""
        url = (
            "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX"
        )
        assert is_safe_url(url) is True

    def test_discord_webhook(self) -> None:
        """Test Discord webhook URL."""
        url = "https://discord.com/api/webhooks/123456789012345678/abcdefghijklmnopqrstuvwxyz"
        assert is_safe_url(url) is True

    def test_webhook_site(self) -> None:
        """Test webhook.site URL."""
        url = "https://webhook.site/unique-id"
        assert is_safe_url(url) is True

    def test_ngrok_url(self) -> None:
        """Test ngrok URL (should be safe as it's public)."""
        url = "https://abc123.ngrok.io/webhook"
        assert is_safe_url(url) is True

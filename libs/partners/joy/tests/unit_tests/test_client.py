"""Tests for Joy Trust client."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_joy.client import JoyTrustClient, JoyTrustError


class TestJoyTrustClient:
    """Tests for JoyTrustClient."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        client = JoyTrustClient()
        assert client.base_url == "https://choosejoy.com.au"
        assert client.api_key is None
        assert client.timeout == 10.0

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        client = JoyTrustClient(
            api_key="test_key",
            base_url="https://custom.url",
            timeout=5.0,
        )
        assert client.base_url == "https://custom.url"
        assert client.api_key == "test_key"
        assert client.timeout == 5.0

    def test_get_headers_no_key(self) -> None:
        """Test headers without API key."""
        client = JoyTrustClient()
        headers = client._get_headers()
        assert "Content-Type" in headers
        assert "x-api-key" not in headers

    def test_get_headers_with_key(self) -> None:
        """Test headers with API key."""
        client = JoyTrustClient(api_key="test_key")
        headers = client._get_headers()
        assert headers["x-api-key"] == "test_key"

    def test_caching(self) -> None:
        """Test response caching."""
        client = JoyTrustClient(cache_ttl=300)

        # Set cache
        client._set_cached("test_key", {"value": 123})

        # Get cached value
        result = client._get_cached("test_key")
        assert result == {"value": 123}

        # Non-existent key
        result = client._get_cached("missing_key")
        assert result is None

    @patch("langchain_joy.client.httpx.Client")
    def test_get_trust_score_success(self, mock_client_class: MagicMock) -> None:
        """Test successful trust score retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "agent_id": "ag_test",
            "trust_score": 2.5,
            "verified": True,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = JoyTrustClient()
        result = client.get_trust_score("ag_test")

        assert result["trust_score"] == 2.5
        assert result["verified"] is True

    def test_verify_trust_meets_threshold(self) -> None:
        """Test verify_trust when threshold is met."""
        client = JoyTrustClient()
        client.get_trust_score = MagicMock(return_value={  # type: ignore
            "trust_score": 2.0,
            "verified": True,
        })

        result = client.verify_trust("ag_test", min_trust=1.5)

        assert result["meets_threshold"] is True
        assert result["trust_score"] == 2.0

    def test_verify_trust_below_threshold(self) -> None:
        """Test verify_trust when below threshold."""
        client = JoyTrustClient()
        client.get_trust_score = MagicMock(return_value={  # type: ignore
            "trust_score": 1.0,
            "verified": False,
        })

        result = client.verify_trust("ag_test", min_trust=1.5)

        assert result["meets_threshold"] is False
        assert result["trust_score"] == 1.0

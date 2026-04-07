"""Shared test fixtures for langchain-ai-identity."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def fake_config() -> dict[str, str]:
    """Return a minimal configuration dict for testing."""
    return {
        "agent_id": "test-agent-uuid",
        "api_key": "aid_sk_test_key_123",
    }


@pytest.fixture()
def mock_gateway_allow():
    """Return a mock httpx response that allows access."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"decision": "allow"}
    mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.fixture()
def mock_gateway_deny():
    """Return a mock httpx response that denies access."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "decision": "deny",
        "reason": "policy violation",
    }
    mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.fixture()
def mock_audit_success():
    """Return a mock httpx response for a successful audit post."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "ok"}
    mock_response.raise_for_status = MagicMock()
    return mock_response

"""Unit tests for langchain_ai_identity._gateway module."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import httpx
import pytest

from langchain_ai_identity._gateway import enforce_access, post_audit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(response: MagicMock) -> MagicMock:
    """Build a mock ``httpx.Client`` context-manager that returns *response*."""
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.return_value = response
    return mock_client


def _make_response(json_body: dict) -> MagicMock:
    """Build a mock httpx response."""
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# enforce_access – allow
# ---------------------------------------------------------------------------


class TestEnforceAccessAllow:
    def test_returns_allow_decision(self) -> None:
        response = _make_response({"decision": "allow"})
        client = _make_mock_client(response)

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            result = enforce_access(
                api_key="aid_sk_test",
                agent_id="test-uuid",
                endpoint="/v1/chat/completions",
            )

        assert result["decision"] == "allow"

    def test_posts_to_gateway_enforce_path(self) -> None:
        response = _make_response({"decision": "allow"})
        client = _make_mock_client(response)

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            enforce_access(
                api_key="aid_sk_test",
                agent_id="test-uuid",
                endpoint="/v1/chat/completions",
            )

        call_args = client.post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "/gateway/enforce" in url


# ---------------------------------------------------------------------------
# enforce_access – deny
# ---------------------------------------------------------------------------


class TestEnforceAccessDeny:
    def test_deny_fail_closed_raises(self) -> None:
        response = _make_response({"decision": "deny", "reason": "policy violation"})
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "denied",
            request=MagicMock(),
            response=MagicMock(status_code=403),
        )

        client = _make_mock_client(response)
        client.post.return_value = response

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            with pytest.raises(PermissionError):
                enforce_access(
                    api_key="aid_sk_test",
                    agent_id="test-uuid",
                    endpoint="/v1/chat/completions",
                    fail_closed=True,
                )

    def test_deny_fail_open_returns_allow(self) -> None:
        response = _make_response({"decision": "deny", "reason": "policy violation"})
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "denied",
            request=MagicMock(),
            response=MagicMock(status_code=403),
        )

        client = _make_mock_client(response)

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = enforce_access(
                    api_key="aid_sk_test",
                    agent_id="test-uuid",
                    endpoint="/v1/chat/completions",
                    fail_closed=False,
                )

        assert result["decision"] == "allow"
        assert "fail-open" in result.get("reason", "")
        assert len(w) >= 1


# ---------------------------------------------------------------------------
# enforce_access – network errors
# ---------------------------------------------------------------------------


class TestEnforceAccessNetworkError:
    def test_network_error_fail_closed_raises(self) -> None:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.post.side_effect = httpx.ConnectError("connection refused")

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            with pytest.raises(RuntimeError, match="Gateway unreachable"):
                enforce_access(
                    api_key="aid_sk_test",
                    agent_id="test-uuid",
                    endpoint="/v1/chat/completions",
                    fail_closed=True,
                )

    def test_network_error_fail_open_returns_allow(self) -> None:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.post.side_effect = httpx.ConnectError("connection refused")

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = enforce_access(
                    api_key="aid_sk_test",
                    agent_id="test-uuid",
                    endpoint="/v1/chat/completions",
                    fail_closed=False,
                )

        assert result["decision"] == "allow"
        assert len(w) >= 1


# ---------------------------------------------------------------------------
# enforce_access – custom gateway URL
# ---------------------------------------------------------------------------


class TestEnforceAccessCustomUrl:
    def test_custom_gateway_url(self) -> None:
        response = _make_response({"decision": "allow"})
        client = _make_mock_client(response)

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            enforce_access(
                api_key="aid_sk_test",
                agent_id="test-uuid",
                endpoint="/v1/chat/completions",
                gateway_url="https://custom-gateway.example.com",
            )

        call_args = client.post.call_args
        url = call_args[0][0]
        assert url.startswith("https://custom-gateway.example.com")

    def test_env_var_gateway_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "AI_IDENTITY_GATEWAY_URL", "https://env-gateway.example.com"
        )

        response = _make_response({"decision": "allow"})
        client = _make_mock_client(response)

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            enforce_access(
                api_key="aid_sk_test",
                agent_id="test-uuid",
                endpoint="/v1/chat/completions",
            )

        call_args = client.post.call_args
        url = call_args[0][0]
        assert url.startswith("https://env-gateway.example.com")


# ---------------------------------------------------------------------------
# post_audit
# ---------------------------------------------------------------------------


class TestPostAudit:
    def test_success(self) -> None:
        response = _make_response({"status": "ok"})
        client = _make_mock_client(response)

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            # Should not raise
            post_audit(
                api_key="aid_sk_test",
                agent_id="test-uuid",
                event_type="llm_start",
                endpoint="/v1/chat/completions",
                decision="allow",
            )

    def test_failure_fail_closed(self) -> None:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.post.side_effect = httpx.ConnectError("connection refused")

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            with pytest.raises(RuntimeError, match="Audit log failed"):
                post_audit(
                    api_key="aid_sk_test",
                    agent_id="test-uuid",
                    event_type="llm_start",
                    endpoint="/v1/chat/completions",
                    decision="allow",
                    fail_closed=True,
                )

    def test_failure_fail_open(self) -> None:
        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.post.side_effect = httpx.ConnectError("connection refused")

        with patch(
            "langchain_ai_identity._gateway.httpx.Client", return_value=client
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                post_audit(
                    api_key="aid_sk_test",
                    agent_id="test-uuid",
                    event_type="llm_start",
                    endpoint="/v1/chat/completions",
                    decision="allow",
                    fail_closed=False,
                )

        assert len(w) >= 1

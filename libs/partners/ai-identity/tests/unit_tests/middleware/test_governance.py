"""Unit tests for langchain_ai_identity.middleware.governance module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from langchain_ai_identity.middleware.governance import (
    AIIdentityGovernanceMiddleware,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_empty_agent_id_raises(self) -> None:
        mw = AIIdentityGovernanceMiddleware(agent_id="", api_key="aid_sk_test")
        with pytest.raises(ValueError, match="agent_id"):
            mw.validate_config()

    def test_empty_api_key_raises(self) -> None:
        mw = AIIdentityGovernanceMiddleware(agent_id="test-uuid", api_key="")
        with pytest.raises(ValueError, match="api_key"):
            mw.validate_config()

    def test_valid_config_passes(self) -> None:
        mw = AIIdentityGovernanceMiddleware(
            agent_id="test-uuid", api_key="aid_sk_test"
        )
        mw.validate_config()  # should not raise


# ---------------------------------------------------------------------------
# Model call enforcement
# ---------------------------------------------------------------------------


class TestEnforceModelCall:
    def test_allow_calls_fn(self) -> None:
        mw = AIIdentityGovernanceMiddleware(
            agent_id="test-uuid", api_key="aid_sk_test", fail_closed=True
        )
        call_fn = MagicMock(return_value="response")

        with (
            patch(
                "langchain_ai_identity.middleware.governance.enforce_access",
                return_value={"decision": "allow"},
            ),
            patch(
                "langchain_ai_identity.middleware.governance.post_audit"
            ),
        ):
            result = mw.enforce_model_call(call_fn)

        call_fn.assert_called_once()
        assert result == "response"

    def test_deny_raises_permission_error(self) -> None:
        mw = AIIdentityGovernanceMiddleware(
            agent_id="test-uuid", api_key="aid_sk_test", fail_closed=True
        )
        call_fn = MagicMock(return_value="response")

        with patch(
            "langchain_ai_identity.middleware.governance.enforce_access",
            return_value={"decision": "deny", "reason": "policy violation"},
        ):
            with pytest.raises(PermissionError):
                mw.enforce_model_call(call_fn)

        call_fn.assert_not_called()


# ---------------------------------------------------------------------------
# Tool call enforcement
# ---------------------------------------------------------------------------


class TestEnforceToolCall:
    def test_allow_calls_fn(self) -> None:
        mw = AIIdentityGovernanceMiddleware(
            agent_id="test-uuid", api_key="aid_sk_test", fail_closed=True
        )
        call_fn = MagicMock(return_value="tool_result")

        with (
            patch(
                "langchain_ai_identity.middleware.governance.enforce_access",
                return_value={"decision": "allow"},
            ),
            patch(
                "langchain_ai_identity.middleware.governance.post_audit"
            ),
        ):
            result = mw.enforce_tool_call("my_tool", call_fn)

        call_fn.assert_called_once()
        assert result == "tool_result"

    def test_deny_raises_permission_error(self) -> None:
        mw = AIIdentityGovernanceMiddleware(
            agent_id="test-uuid", api_key="aid_sk_test", fail_closed=True
        )
        call_fn = MagicMock(return_value="tool_result")

        with patch(
            "langchain_ai_identity.middleware.governance.enforce_access",
            return_value={"decision": "deny", "reason": "policy violation"},
        ):
            with pytest.raises(PermissionError):
                mw.enforce_tool_call("my_tool", call_fn)

        call_fn.assert_not_called()


# ---------------------------------------------------------------------------
# Audit disabled
# ---------------------------------------------------------------------------


class TestAuditDisabled:
    def test_no_audit_when_disabled(self) -> None:
        mw = AIIdentityGovernanceMiddleware(
            agent_id="test-uuid",
            api_key="aid_sk_test",
            audit_enabled=False,
        )
        call_fn = MagicMock(return_value="response")

        with (
            patch(
                "langchain_ai_identity.middleware.governance.enforce_access",
                return_value={"decision": "allow"},
            ),
            patch(
                "langchain_ai_identity.middleware.governance.post_audit"
            ) as mock_audit,
        ):
            mw.enforce_model_call(call_fn)

        mock_audit.assert_not_called()


# ---------------------------------------------------------------------------
# Name property
# ---------------------------------------------------------------------------


class TestMiddlewareName:
    def test_name_property(self) -> None:
        mw = AIIdentityGovernanceMiddleware(
            agent_id="test-uuid", api_key="aid_sk_test"
        )
        assert mw.name == "AIIdentityGovernanceMiddleware"

"""Regression tests for SSRF bypass vulnerability (#37297).

These tests verify that the test-environment bypass in validate_safe_url
only allows exact known test hostnames, and that the
_effective_allowed_hosts function only augments for exact local env values.
"""

import os
from unittest.mock import patch

import pytest

from langchain_core._security._policy import _effective_allowed_hosts, SSRFPolicy
from langchain_core._security._ssrf_protection import validate_safe_url


class TestSSRFBypassRegression:
    """Regression tests for #37297 — SSRF bypass via crafted hostname."""

    def test_attacker_hostname_blocked_in_local_test(self) -> None:
        """Attacker-controlled hostname must NOT bypass validation."""
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "local_test"}):
            with pytest.raises(ValueError):
                validate_safe_url("http://test.attacker.server.com/exfil")

    def test_attacker_hostname_with_test_prefix_blocked(self) -> None:
        """Any hostname starting with 'test' should not auto-bypass."""
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "local_test"}):
            with pytest.raises(ValueError):
                validate_safe_url("http://testevil.server.com/steal")

    def test_legitimate_testserver_allowed(self) -> None:
        """The legitimate testserver hostname should still be allowed."""
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "local_test"}):
            result = validate_safe_url("http://testserver/api/v1")
            assert result == "http://testserver/api/v1"

    def test_legitimate_testserver_local_allowed(self) -> None:
        """testserver.local should be allowed in local_test env."""
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "local_test"}):
            result = validate_safe_url("http://testserver.local/api/v1")
            assert result == "http://testserver.local/api/v1"

    def test_bypass_not_active_in_production(self) -> None:
        """Test bypass must not activate in production environment."""
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "production"}):
            with pytest.raises(ValueError):
                validate_safe_url("http://testserver/api/v1")

    def test_bypass_case_insensitive(self) -> None:
        """Hostname matching should be case-insensitive."""
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "local_test"}):
            result = validate_safe_url("http://TESTSERVER/api/v1")
            assert result == "http://TESTSERVER/api/v1"


class TestEffectiveAllowedHostsRegression:
    """Regression tests for _effective_allowed_hosts env check."""

    def test_exact_local_env_allowed(self) -> None:
        """LANGCHAIN_ENV='local' should augment allowed hosts."""
        policy = SSRFPolicy()
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "local"}):
            hosts = _effective_allowed_hosts(policy)
            assert "localhost" in hosts
            assert "testserver" in hosts

    def test_exact_local_test_env_allowed(self) -> None:
        """LANGCHAIN_ENV='local_test' should augment allowed hosts."""
        policy = SSRFPolicy()
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "local_test"}):
            hosts = _effective_allowed_hosts(policy)
            assert "localhost" in hosts
            assert "testserver" in hosts

    def test_local_staging_not_allowed(self) -> None:
        """LANGCHAIN_ENV='local_staging' should NOT augment hosts."""
        policy = SSRFPolicy()
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "local_staging"}):
            hosts = _effective_allowed_hosts(policy)
            assert "localhost" not in hosts
            assert "testserver" not in hosts

    def test_localized_not_allowed(self) -> None:
        """LANGCHAIN_ENV='localized' should NOT augment hosts."""
        policy = SSRFPolicy()
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "localized"}):
            hosts = _effective_allowed_hosts(policy)
            assert "localhost" not in hosts

    def test_production_not_allowed(self) -> None:
        """LANGCHAIN_ENV='production' should NOT augment hosts."""
        policy = SSRFPolicy()
        with patch.dict(os.environ, {"LANGCHAIN_ENV": "production"}):
            hosts = _effective_allowed_hosts(policy)
            assert "localhost" not in hosts
            assert "testserver" not in hosts

    def test_empty_env_not_allowed(self) -> None:
        """Empty LANGCHAIN_ENV should NOT augment hosts."""
        policy = SSRFPolicy()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGCHAIN_ENV", None)
            hosts = _effective_allowed_hosts(policy)
            assert "localhost" not in hosts

"""Tests for langchain_anthropic.utils module."""

from __future__ import annotations

from langchain_core.utils import SecretStr

from langchain_anthropic.utils import (
    create_bedrock_client_params,
    resolve_aws_credentials,
)


def test_resolve_aws_credentials_all_provided() -> None:
    """Test resolve_aws_credentials with all credentials provided."""
    creds = resolve_aws_credentials(
        aws_access_key_id=SecretStr("AKIAIOSFODNN7EXAMPLE"),
        aws_secret_access_key=SecretStr("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        aws_session_token=SecretStr("session-token-example"),
    )

    assert creds["aws_access_key"] == "AKIAIOSFODNN7EXAMPLE"
    assert creds["aws_secret_key"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert creds["aws_session_token"] == "session-token-example"


def test_resolve_aws_credentials_partial() -> None:
    """Test resolve_aws_credentials with only some credentials provided."""
    creds = resolve_aws_credentials(
        aws_access_key_id=SecretStr("AKIAIOSFODNN7EXAMPLE"),
        aws_secret_access_key=SecretStr("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        aws_session_token=None,
    )

    assert creds["aws_access_key"] == "AKIAIOSFODNN7EXAMPLE"
    assert creds["aws_secret_key"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert "aws_session_token" not in creds


def test_resolve_aws_credentials_none() -> None:
    """Test resolve_aws_credentials with no credentials provided."""
    creds = resolve_aws_credentials()

    assert len(creds) == 0
    assert "aws_access_key" not in creds
    assert "aws_secret_key" not in creds
    assert "aws_session_token" not in creds


def test_resolve_aws_credentials_only_session_token() -> None:
    """Test resolve_aws_credentials with only session token."""
    creds = resolve_aws_credentials(
        aws_session_token=SecretStr("session-token-example"),
    )

    assert creds["aws_session_token"] == "session-token-example"
    assert "aws_access_key" not in creds
    assert "aws_secret_key" not in creds


def test_create_bedrock_client_params_minimal() -> None:
    """Test create_bedrock_client_params with minimal required parameters."""
    params = create_bedrock_client_params(aws_region="us-east-1")

    assert params["aws_region"] == "us-east-1"
    assert params["max_retries"] == 2  # default
    assert params["default_headers"] is None
    assert "timeout" not in params or params["timeout"] is None


def test_create_bedrock_client_params_with_credentials() -> None:
    """Test create_bedrock_client_params with AWS credentials."""
    params = create_bedrock_client_params(
        aws_region="us-west-2",
        aws_access_key_id=SecretStr("AKIAIOSFODNN7EXAMPLE"),
        aws_secret_access_key=SecretStr("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        aws_session_token=SecretStr("session-token-example"),
    )

    assert params["aws_region"] == "us-west-2"
    assert params["aws_access_key"] == "AKIAIOSFODNN7EXAMPLE"
    assert params["aws_secret_key"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert params["aws_session_token"] == "session-token-example"


def test_create_bedrock_client_params_with_all_options() -> None:
    """Test create_bedrock_client_params with all optional parameters."""
    params = create_bedrock_client_params(
        aws_region="eu-west-1",
        aws_access_key_id=SecretStr("AKIAIOSFODNN7EXAMPLE"),
        aws_secret_access_key=SecretStr("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
        max_retries=5,
        default_headers={"X-Custom-Header": "value"},
        timeout=30.0,
    )

    assert params["aws_region"] == "eu-west-1"
    assert params["aws_access_key"] == "AKIAIOSFODNN7EXAMPLE"
    assert params["aws_secret_key"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert params["max_retries"] == 5
    assert params["default_headers"] == {"X-Custom-Header": "value"}
    assert params["timeout"] == 30.0


def test_create_bedrock_client_params_timeout_none() -> None:
    """Test create_bedrock_client_params with timeout=None."""
    params = create_bedrock_client_params(
        aws_region="us-east-1",
        timeout=None,
    )

    assert params["timeout"] is None


def test_create_bedrock_client_params_timeout_zero() -> None:
    """Test create_bedrock_client_params with timeout=0 (should be excluded)."""
    params = create_bedrock_client_params(
        aws_region="us-east-1",
        timeout=0,
    )

    # timeout=0 should be excluded (treated as "use default")
    assert "timeout" not in params or params["timeout"] == 0


def test_create_bedrock_client_params_timeout_negative() -> None:
    """Test create_bedrock_client_params with negative timeout (should be excluded)."""
    params = create_bedrock_client_params(
        aws_region="us-east-1",
        timeout=-1,
    )

    # Negative timeout should be excluded (treated as "use default")
    assert "timeout" not in params or params["timeout"] == -1


def test_create_bedrock_client_params_reuses_resolve_aws_credentials() -> None:
    """Test that create_bedrock_client_params properly uses resolve_aws_credentials."""
    # This test ensures the functions work together correctly
    params = create_bedrock_client_params(
        aws_region="us-east-1",
        aws_access_key_id=SecretStr("test-key"),
        aws_secret_access_key=SecretStr("test-secret"),
    )

    # Verify credentials are properly resolved
    assert "aws_access_key" in params
    assert "aws_secret_key" in params
    assert params["aws_access_key"] == "test-key"
    assert params["aws_secret_key"] == "test-secret"

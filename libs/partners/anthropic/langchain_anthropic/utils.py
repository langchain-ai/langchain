"""Shared utilities for Anthropic integrations.

This module provides shared helpers for AWS credential resolution and Bedrock
client creation, used by ChatAnthropicBedrock and other Bedrock-based integrations.
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr


def resolve_aws_credentials(
    aws_access_key_id: SecretStr | None = None,
    aws_secret_access_key: SecretStr | None = None,
    aws_session_token: SecretStr | None = None,
) -> dict[str, Any]:
    """Resolve AWS credentials for Bedrock client initialization.

    Extracts secret values from SecretStr fields, only including credentials
    that are provided. This allows the AnthropicBedrock client to fall back
    to boto3's default credential chain when credentials are not explicitly
    provided.

    Args:
        aws_access_key_id: Optional AWS access key ID as SecretStr.
        aws_secret_access_key: Optional AWS secret access key as SecretStr.
        aws_session_token: Optional AWS session token as SecretStr.

    Returns:
        Dictionary with AWS credential parameters. Keys are:
        - `aws_access_key`: Access key ID value (if provided)
        - `aws_secret_key`: Secret access key value (if provided)
        - `aws_session_token`: Session token value (if provided)

    Example:
        ```python
        from langchain_anthropic.utils import resolve_aws_credentials
        from pydantic import SecretStr

        creds = resolve_aws_credentials(
            aws_access_key_id=SecretStr("AKIA..."),
            aws_secret_access_key=SecretStr("secret..."),
        )
        # Returns: {"aws_access_key": "AKIA...", "aws_secret_key": "secret..."}
        ```
    """
    credentials: dict[str, Any] = {}

    if aws_access_key_id:
        credentials["aws_access_key"] = aws_access_key_id.get_secret_value()
    if aws_secret_access_key:
        credentials["aws_secret_key"] = aws_secret_access_key.get_secret_value()
    if aws_session_token:
        credentials["aws_session_token"] = aws_session_token.get_secret_value()

    return credentials


def create_bedrock_client_params(
    aws_region: str,
    aws_access_key_id: SecretStr | None = None,
    aws_secret_access_key: SecretStr | None = None,
    aws_session_token: SecretStr | None = None,
    max_retries: int = 2,
    default_headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Create client parameters for AnthropicBedrock client initialization.

    Builds a complete parameter dictionary for initializing AnthropicBedrock
    or AsyncAnthropicBedrock clients with AWS credentials and configuration.

    Args:
        aws_region: AWS region for Bedrock API calls (e.g., "us-east-1").
        aws_access_key_id: Optional AWS access key ID as SecretStr.
        aws_secret_access_key: Optional AWS secret access key as SecretStr.
        aws_session_token: Optional AWS session token as SecretStr.
        max_retries: Maximum number of retry attempts for requests.
        default_headers: Optional default headers to include in requests.
        timeout: Optional timeout in seconds for requests. None or values <= 0
            are treated as "use default".

    Returns:
        Dictionary of parameters ready to pass to AnthropicBedrock or
        AsyncAnthropicBedrock constructor.

    Example:
        ```python
        from langchain_anthropic.utils import create_bedrock_client_params
        from pydantic import SecretStr
        from anthropic import AnthropicBedrock

        params = create_bedrock_client_params(
            aws_region="us-east-1",
            aws_access_key_id=SecretStr("AKIA..."),
            aws_secret_access_key=SecretStr("secret..."),
            max_retries=3,
            timeout=30.0,
        )
        client = AnthropicBedrock(**params)
        ```
    """
    client_params: dict[str, Any] = {
        "aws_region": aws_region,
        "max_retries": max_retries,
        "default_headers": (default_headers or None),
    }

    # Resolve and add AWS credentials
    credentials = resolve_aws_credentials(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )
    client_params.update(credentials)

    # Handle timeout: None or values <= 0 indicate "use default"
    # None is a meaningful value for Anthropic client and treated differently
    # than not specifying the param at all
    if timeout is None or timeout > 0:
        client_params["timeout"] = timeout

    return client_params

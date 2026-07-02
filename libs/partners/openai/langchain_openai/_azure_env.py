"""Azure OpenAI environment variable helpers."""

from __future__ import annotations

import os
import warnings

from pydantic import SecretStr


def azure_openai_api_key_from_env() -> SecretStr | None:
    """Resolve the Azure OpenAI API key from environment variables.

    Prefers ``AZURE_OPENAI_API_KEY``. Falls back to ``OPENAI_API_KEY`` with a
    deprecation warning for backwards compatibility.

    Returns:
        The API key if found, otherwise ``None``.
    """
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if azure_key:
        return SecretStr(azure_key)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        warnings.warn(
            "Using OPENAI_API_KEY for Azure OpenAI is deprecated. "
            "Set AZURE_OPENAI_API_KEY instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SecretStr(openai_key)

    return None

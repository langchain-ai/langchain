"""Shared configuration for LangSmith Gateway provider endpoints."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final, Literal, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import NotRequired


GatewayProvider = Literal["anthropic", "fireworks", "google_genai", "openai"]


class GatewayProviderConfig(TypedDict):
    """LangSmith Gateway capabilities and URL path for a model provider."""

    path: str
    oauth: bool
    static_key: bool
    notes: NotRequired[str]


LANGSMITH_GATEWAY_URL: Final = "https://gateway.smith.langchain.com"
"""Default LangSmith Gateway root URL."""


LANGSMITH_GATEWAY_PROVIDERS: Final[Mapping[GatewayProvider, GatewayProviderConfig]] = {
    "anthropic": {"path": "/anthropic", "oauth": True, "static_key": True},
    "fireworks": {"path": "/fireworks", "oauth": True, "static_key": True},
    "google_genai": {
        "path": "/google-genai",
        "oauth": False,
        "static_key": False,
        "notes": "Not supported by LangSmith Gateway.",
    },
    "openai": {"path": "/openai/v1", "oauth": True, "static_key": True},
}
"""Initial LangSmith Gateway provider support matrix.

Each entry declares the path appended to a Gateway root and whether the provider
is available through OAuth or a static Gateway API key. `google_genai` is
included to make its current unsupported status explicit.
"""


_DISABLED_GATEWAY_VALUES: Final = frozenset({"false", "0", "no"})
_ENABLED_GATEWAY_VALUES: Final = frozenset({"true", "1", "yes"})


def resolve_langsmith_gateway_url(provider: GatewayProvider) -> str | None:
    """Resolve the configured LangSmith Gateway URL for a provider.

    `LANGSMITH_GATEWAY=true` selects the hosted Gateway. A custom
    `LANGSMITH_GATEWAY` value may be either a Gateway root URL, in which case
    the provider path is appended, or a provider-qualified URL, which is used
    unchanged. Disabled and unset values return `None`.

    Args:
        provider: Provider whose Gateway endpoint should be resolved.

    Returns:
        A provider-qualified Gateway URL, or `None` when Gateway is disabled.

    Raises:
        ValueError: If the provider is not supported by LangSmith Gateway.
    """
    raw_url = os.getenv("LANGSMITH_GATEWAY")
    if raw_url is None or raw_url.lower() in _DISABLED_GATEWAY_VALUES:
        return None

    provider_config = LANGSMITH_GATEWAY_PROVIDERS[provider]
    if not provider_config["oauth"] and not provider_config["static_key"]:
        msg = f"LangSmith Gateway does not support provider `{provider}`."
        raise ValueError(msg)

    provider_path = provider_config["path"]
    if raw_url.lower() not in _ENABLED_GATEWAY_VALUES:
        if raw_url.rstrip("/").endswith(provider_path):
            return raw_url
        return f"{raw_url.rstrip('/')}{provider_path}"

    gateway_url = LANGSMITH_GATEWAY_URL
    if gateway_url.endswith(provider_path):
        return gateway_url
    return f"{gateway_url}{provider_path}"

"""Private helpers for resolving LangSmith gateway configuration.

The [LangSmith LLM gateway](https://docs.langchain.com/langsmith/llm-gateway)
lets a chat model reach a provider through a proxy configured via environment
variables. These helpers centralize the (non-trivial) precedence rules so that
each provider integration resolves its base URL and API key identically.

This module is private: the API is not stable and may change without notice.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import SecretStr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

_LANGSMITH_GATEWAY_ENV = "LANGSMITH_GATEWAY"
_LANGSMITH_GATEWAY_API_KEY_ENV = "LANGSMITH_GATEWAY_API_KEY"
_LANGSMITH_GATEWAY_DEFAULT_BASE = "https://gateway.smith.langchain.com"

_TRUE_VALUES = ("true", "1", "yes")
_FALSE_VALUES = ("false", "0", "no")


class GatewayConfig(NamedTuple):
    """Resolved gateway configuration.

    Attributes:
        base_url: The base URL the client should use, or None to defer to the
            provider SDK's own default.
        api_key: The API key to use. A ``SecretStr`` when derived from the
            environment; the caller's value returned unchanged when one was
            passed explicitly; None when no key could be resolved.
        base_url_from_gateway: Whether ``base_url`` was populated from the
            LangSmith gateway (as opposed to an explicit value, a provider env
            var, or a default).
    """

    base_url: str | None
    api_key: Any
    base_url_from_gateway: bool


def _first_env(names: str | Sequence[str]) -> str | None:
    """Return the first non-empty value among ``names``, or None."""
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _resolve_gateway_base_url(provider_path: str) -> str | None:
    """Resolve the LangSmith gateway base URL for a provider.

    ``LANGSMITH_GATEWAY`` accepts either a boolean-ish string or an explicit
    base URL:

    - ``true`` / ``1`` / ``yes`` -> the default gateway host.
    - ``false`` / ``0`` / ``no`` / unset -> the gateway is disabled (None).
    - anything else -> treated as a custom gateway base URL.

    The provider-specific path is appended in all enabled cases.

    Args:
        provider_path: Path segment for the provider, e.g. ``"openai/v1"`` or
            ``"anthropic"``.

    Returns:
        The provider base URL on the gateway, or None if the gateway is
        disabled.
    """
    raw = os.getenv(_LANGSMITH_GATEWAY_ENV)
    if raw is None or raw.lower() in _FALSE_VALUES:
        return None
    base = (
        _LANGSMITH_GATEWAY_DEFAULT_BASE
        if raw.lower() in _TRUE_VALUES
        else raw.rstrip("/")
    )
    return f"{base}/{provider_path}"


def _resolve_gateway_config(
    *,
    base_url: str | None,
    api_key: Any,
    provider_path: str,
    base_url_env: str | Sequence[str] = (),
    api_key_env: str | Sequence[str] = (),
    default_base_url: str | None = None,
) -> GatewayConfig:
    """Resolve a provider's base URL and API key, applying gateway settings.

    Precedence:

    - **base_url:** explicit ``base_url`` > ``base_url_env`` > LangSmith gateway
      > ``default_base_url``.
    - **api_key:** explicit ``api_key`` (returned unchanged) > the gateway key if
      the base URL came from the gateway, otherwise the provider key > the other.
      The gateway key is only a candidate when the gateway is enabled.

    The provenance flip on the key means an ``OPENAI_API_KEY``-style provider key
    is preferred whenever the caller pointed the base URL at a non-gateway
    endpoint (so a stray gateway key is not sent to the provider), while the
    gateway key wins for the common "just enable the gateway" setup even if a
    provider key happens to be present in the environment.

    Args:
        base_url: Explicitly-provided base URL (e.g. a ``base_url`` kwarg), or
            None if not set by the caller.
        api_key: Explicitly-provided API key, or None if not set by the caller.
            Returned unchanged when not None, so a caller-supplied value (secret
            or callable) always wins.
        provider_path: Path segment appended to the gateway host.
        base_url_env: Env var name(s) for the provider base URL, in priority
            order.
        api_key_env: Env var name(s) for the provider API key, in priority order.
        default_base_url: Base URL used when nothing else is set.

    Returns:
        The resolved `GatewayConfig`.
    """
    gateway_base_url = _resolve_gateway_base_url(provider_path)

    resolved_base_url = base_url
    base_url_from_gateway = False
    if resolved_base_url is None:
        resolved_base_url = _first_env(base_url_env)
        if resolved_base_url is None:
            if gateway_base_url is not None:
                resolved_base_url = gateway_base_url
                base_url_from_gateway = True
            else:
                resolved_base_url = default_base_url

    if api_key is not None:
        resolved_api_key: Any = api_key
    else:
        gateway_api_key = (
            os.getenv(_LANGSMITH_GATEWAY_API_KEY_ENV)
            if gateway_base_url is not None
            else None
        )
        provider_api_key = _first_env(api_key_env)
        chosen = (
            (gateway_api_key or provider_api_key)
            if base_url_from_gateway
            else (provider_api_key or gateway_api_key)
        )
        resolved_api_key = SecretStr(chosen) if chosen else None

    return GatewayConfig(resolved_base_url, resolved_api_key, base_url_from_gateway)


def _pop_provided(values: dict[str, Any], cls: type[BaseModel], field: str) -> Any:
    """Pop a caller-provided field value by name or alias; None if absent.

    Handles models with ``populate_by_name=True``, where a field may be supplied
    under either its name or its alias. Both keys are removed so the resolved
    value can be written back canonically under the field name. The alias is read
    from the model rather than hard-coded, so callers pass only field names.
    """
    value = values.pop(field, None)
    alias = cls.model_fields[field].alias
    if alias is not None:
        alias_value = values.pop(alias, None)
        if value is None:
            value = alias_value
    return value


def _apply_gateway_config(
    values: dict[str, Any],
    cls: type[BaseModel],
    *,
    base_url_field: str,
    api_key_field: str,
    provider_path: str,
    base_url_env: str | Sequence[str] = (),
    api_key_env: str | Sequence[str] = (),
    default_base_url: str | None = None,
) -> GatewayConfig:
    """Resolve gateway settings from a model's raw input in a "before" validator.

    Reads the caller-provided base URL and API key (by field name or alias),
    resolves them against the gateway and provider env vars via
    `_resolve_gateway_config`, then writes the results back into ``values`` under
    the canonical field names. This lets a provider integration keep a
    non-optional key field: the resolved value is injected before field
    validation runs, so the field always receives a concrete value.

    The resolved API key is written only when non-None, so the field's own
    default applies when no key is found. Returns the `GatewayConfig` so the
    caller can, for example, raise when a required key is missing.

    Args:
        values: The raw input mapping passed to the model, mutated in place.
        cls: The model class, used to look up field aliases.
        base_url_field: Name of the base URL field (e.g. ``"anthropic_api_url"``).
        api_key_field: Name of the API key field (e.g. ``"anthropic_api_key"``).
        provider_path: Path segment appended to the gateway host.
        base_url_env: Env var name(s) for the provider base URL.
        api_key_env: Env var name(s) for the provider API key.
        default_base_url: Base URL used when nothing else is set.

    Returns:
        The resolved `GatewayConfig`.
    """
    config = _resolve_gateway_config(
        base_url=_pop_provided(values, cls, base_url_field),
        api_key=_pop_provided(values, cls, api_key_field),
        provider_path=provider_path,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
        default_base_url=default_base_url,
    )
    values[base_url_field] = config.base_url
    if config.api_key is not None:
        values[api_key_field] = config.api_key
    return config

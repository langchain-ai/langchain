"""Tests for LangSmith gateway configuration resolution."""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_core.utils._gateway import (
    GatewayConfig,
    _apply_gateway_config,
    _resolve_gateway_base_url,
    _resolve_gateway_config,
)

# A representative provider path; the helper is provider-agnostic.
_PATH = "openai/v1"
_BASE_ENV = "OPENAI_API_BASE"
_KEY_ENV = "OPENAI_API_KEY"
_DEFAULT_GATEWAY = "https://gateway.smith.langchain.com/openai/v1"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a hermetic environment for every test."""
    for var in (
        "LANGSMITH_GATEWAY",
        "LANGSMITH_GATEWAY_API_KEY",
        _BASE_ENV,
        _KEY_ENV,
    ):
        monkeypatch.delenv(var, raising=False)


def _config(**overrides: object) -> GatewayConfig:
    kwargs: dict[str, Any] = {
        "base_url": None,
        "api_key": None,
        "provider_path": _PATH,
        "base_url_env": _BASE_ENV,
        "api_key_env": _KEY_ENV,
    }
    kwargs.update(overrides)
    return _resolve_gateway_config(**kwargs)


# --- _resolve_gateway_base_url --------------------------------------------


@pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE", "Yes"])
def test_base_url_truthy(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", value)
    assert _resolve_gateway_base_url(_PATH) == _DEFAULT_GATEWAY


@pytest.mark.parametrize("value", ["false", "0", "no", "FALSE"])
def test_base_url_falsy(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", value)
    assert _resolve_gateway_base_url(_PATH) is None


def test_base_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_GATEWAY", raising=False)
    assert _resolve_gateway_base_url(_PATH) is None


def test_base_url_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", "https://eu.gateway.example.com/")
    assert (
        _resolve_gateway_base_url(_PATH) == "https://eu.gateway.example.com/openai/v1"
    )


# --- api key + base url resolution matrix ---------------------------------
# Each case mirrors a documented requirement scenario.


def test_gateway_on_with_gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    config = _config()
    assert config.base_url == _DEFAULT_GATEWAY
    assert config.base_url_from_gateway is True
    assert isinstance(config.api_key, SecretStr)
    assert config.api_key.get_secret_value() == "gateway-key"


def test_gateway_key_ignored_when_gateway_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Gateway key set but LANGSMITH_GATEWAY unset -> never used.
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setenv(_KEY_ENV, "provider-key")
    config = _config()
    assert config.base_url is None
    assert config.base_url_from_gateway is False
    assert isinstance(config.api_key, SecretStr)
    assert config.api_key.get_secret_value() == "provider-key"


def test_gateway_off_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", "false")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    config = _config()
    assert config.base_url is None
    assert config.api_key is None


def test_gateway_key_beats_provider_key_on_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setenv(_KEY_ENV, "provider-key")
    config = _config()
    assert config.base_url == _DEFAULT_GATEWAY
    assert config.api_key.get_secret_value() == "gateway-key"


def test_gateway_on_falls_back_to_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv(_KEY_ENV, "provider-key")
    config = _config()
    assert config.base_url == _DEFAULT_GATEWAY
    assert config.api_key.get_secret_value() == "provider-key"


def test_provider_base_url_uses_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Base URL overridden away from the gateway -> provider key wins.
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setenv(_BASE_ENV, "https://api.openai.com/v1")
    monkeypatch.setenv(_KEY_ENV, "provider-key")
    config = _config()
    assert config.base_url == "https://api.openai.com/v1"
    assert config.base_url_from_gateway is False
    assert config.api_key.get_secret_value() == "provider-key"


def test_provider_base_url_falls_back_to_gateway_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Base URL overridden, no provider key -> gateway key used as fallback.
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setenv(_BASE_ENV, "https://api.openai.com/v1")
    config = _config()
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key.get_secret_value() == "gateway-key"


def test_custom_gateway_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", "https://eu.gateway.example.com")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    config = _config()
    assert config.base_url == "https://eu.gateway.example.com/openai/v1"
    assert config.base_url_from_gateway is True
    assert config.api_key.get_secret_value() == "gateway-key"


def test_explicit_kwargs_override_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", "https://eu.gateway.example.com")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setenv(_KEY_ENV, "provider-key")
    explicit_key = SecretStr("explicit-key")
    config = _config(
        base_url="https://apac.gateway.example.com",
        api_key=explicit_key,
    )
    assert config.base_url == "https://apac.gateway.example.com"
    assert config.base_url_from_gateway is False
    assert config.api_key is explicit_key


def test_explicit_base_url_pairs_with_gateway_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Explicit base URL, no explicit key: gateway key still resolved from env.
    monkeypatch.setenv("LANGSMITH_GATEWAY", "https://eu.gateway.example.com")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    config = _config(base_url="https://apac.gateway.example.com")
    assert config.base_url == "https://apac.gateway.example.com"
    assert config.api_key.get_secret_value() == "gateway-key"


def test_no_gateway_uses_provider_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_BASE_ENV, "https://provider.example.com")
    monkeypatch.setenv(_KEY_ENV, "provider-key")
    config = _config(default_base_url="https://default.example.com")
    assert config.base_url == "https://provider.example.com"
    assert config.api_key.get_secret_value() == "provider-key"


def test_default_base_url_when_nothing_set() -> None:
    config = _config(default_base_url="https://default.example.com")
    assert config.base_url == "https://default.example.com"
    assert config.base_url_from_gateway is False
    assert config.api_key is None


def test_multiple_base_url_env_vars_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SECOND_BASE", "https://second.example.com")
    config = _config(base_url_env=["FIRST_BASE", "SECOND_BASE"])
    assert config.base_url == "https://second.example.com"


# --- _apply_gateway_config (before-validator path) ------------------------


class _Model(BaseModel):
    """Stand-in for a provider chat model with aliased fields."""

    model_config = ConfigDict(populate_by_name=True)

    api_base: str | None = Field(default=None, alias="base_url")
    api_key: SecretStr = Field(default=SecretStr(""), alias="api_key_alias")


def _apply(values: dict[str, Any]) -> tuple[dict[str, Any], GatewayConfig]:
    config = _apply_gateway_config(
        values,
        _Model,
        base_url_field="api_base",
        api_key_field="api_key",
        provider_path="openai/v1",
        base_url_env=_BASE_ENV,
        api_key_env=_KEY_ENV,
    )
    return values, config


def test_apply_writes_resolved_values_by_field_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    values, config = _apply({})
    assert values["api_base"] == _DEFAULT_GATEWAY
    assert isinstance(values["api_key"], SecretStr)
    assert values["api_key"].get_secret_value() == "gateway-key"
    assert config.base_url_from_gateway is True


def test_apply_reads_explicit_via_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    # Values provided under aliases are consumed and rewritten under field names.
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    values, _ = _apply(
        {"base_url": "https://explicit.example.com", "api_key_alias": "explicit-key"}
    )
    assert "base_url" not in values
    assert "api_key_alias" not in values
    assert values["api_base"] == "https://explicit.example.com"
    assert values["api_key"] == "explicit-key"


def test_apply_reads_explicit_via_field_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    values, _ = _apply({"api_base": "https://explicit.example.com"})
    assert values["api_base"] == "https://explicit.example.com"


def test_apply_omits_key_when_unresolved() -> None:
    # No key anywhere: the field is left unset so its own default applies.
    values, config = _apply({})
    assert "api_key" not in values
    assert config.api_key is None

"""Unit tests for `langchain_openai.chatgpt_oauth`."""
# ruff: noqa: S105, S106

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from langchain_openai.chatgpt_oauth import (
    CHATGPT_AUTH_CLAIMS_NAMESPACE,
    ChatGPTToken,
    FileChatGPTOAuthTokenProvider,
    _build_authorize_url,
    _generate_pkce_pair,
    _serialize_token,
    _token_from_response,
    decode_jwt_claims,
)


def _make_jwt(payload: dict[str, Any]) -> str:
    """Build an unsigned JWT for tests."""

    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    header = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    body = b64(json.dumps(payload).encode())
    sig = b64(b"sig")
    return f"{header}.{body}.{sig}"


def test_decode_jwt_claims_extracts_namespaced_chatgpt_claims() -> None:
    jwt = _make_jwt(
        {
            "sub": "user-1",
            CHATGPT_AUTH_CLAIMS_NAMESPACE: {
                "chatgpt_account_id": "acct-123",
                "chatgpt_plan_type": "plus",
                "chatgpt_user_id": "user-1",
            },
        }
    )
    claims = decode_jwt_claims(jwt)
    assert claims["sub"] == "user-1"
    auth = claims[CHATGPT_AUTH_CLAIMS_NAMESPACE]
    assert auth["chatgpt_account_id"] == "acct-123"
    assert auth["chatgpt_plan_type"] == "plus"


def test_decode_jwt_claims_handles_malformed_input() -> None:
    assert decode_jwt_claims("") == {}
    assert decode_jwt_claims("not-a-jwt") == {}
    assert decode_jwt_claims("a.b") == {}


def test_token_from_response_extracts_claims_and_falls_back_to_existing_refresh() -> (
    None
):
    id_token = _make_jwt(
        {
            CHATGPT_AUTH_CLAIMS_NAMESPACE: {
                "chatgpt_account_id": "acct-9",
                "chatgpt_plan_type": "pro",
                "chatgpt_user_id": "user-9",
            }
        }
    )
    response = {
        "access_token": "new-at",
        "expires_in": 3600,
        "id_token": id_token,
        # No refresh_token returned: must fall back to existing.
    }
    token = _token_from_response(response, fallback_refresh_token="old-rt")
    assert token.access_token == "new-at"
    assert token.refresh_token == "old-rt"
    assert token.account_id == "acct-9"
    assert token.plan_type == "pro"
    assert token.user_id == "user-9"
    assert token.id_token == id_token
    # expires_at is in the future
    assert token.expires_at > datetime.now(timezone.utc)


def test_token_is_expired_uses_skew() -> None:
    now = datetime.now(timezone.utc)
    token = ChatGPTToken(
        access_token="x",
        refresh_token="y",
        expires_at=now + timedelta(minutes=1),
    )
    assert token.is_expired(skew=timedelta(minutes=5)) is True
    assert token.is_expired(skew=timedelta(seconds=0)) is False


def test_file_provider_persists_token_with_private_perms(tmp_path: Path) -> None:
    store = tmp_path / "chatgpt-auth.json"
    provider = FileChatGPTOAuthTokenProvider(path=store)
    token = ChatGPTToken(
        access_token="at",
        refresh_token="rt",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        account_id="acct-1",
        plan_type="plus",
    )
    provider.save(token)

    assert store.exists()
    if os.name != "nt":
        mode = store.stat().st_mode & 0o777
        assert mode == 0o600

    raw = json.loads(store.read_text())
    assert raw["access_token"] == "at"
    assert raw["account_id"] == "acct-1"

    fresh = FileChatGPTOAuthTokenProvider(path=store)
    reloaded = fresh.get_token()
    assert reloaded.access_token == "at"
    assert reloaded.account_id == "acct-1"


def test_file_provider_get_token_does_not_refresh_when_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "auth.json"
    provider = FileChatGPTOAuthTokenProvider(path=store)
    valid_token = ChatGPTToken(
        access_token="at",
        refresh_token="rt",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    provider.save(valid_token)

    def _explode(*args: Any, **kwargs: Any) -> dict[str, Any]:
        msg = "should not refresh"
        raise AssertionError(msg)

    monkeypatch.setattr("langchain_openai.chatgpt_oauth._post_form", _explode)
    out = provider.get_token()
    assert out.access_token == "at"


def test_file_provider_get_token_refreshes_when_expired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "auth.json"
    provider = FileChatGPTOAuthTokenProvider(path=store)
    expired = ChatGPTToken(
        access_token="old-at",
        refresh_token="old-rt",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    provider.save(expired)

    calls: list[dict[str, Any]] = []

    new_id_token = _make_jwt(
        {CHATGPT_AUTH_CLAIMS_NAMESPACE: {"chatgpt_account_id": "acct-after-refresh"}}
    )

    def _fake_post(url: str, data: dict[str, str], **_: Any) -> dict[str, Any]:
        calls.append({"url": url, "data": data})
        return {
            "access_token": "new-at",
            "expires_in": 3600,
            "id_token": new_id_token,
        }

    monkeypatch.setattr("langchain_openai.chatgpt_oauth._post_form", _fake_post)

    refreshed = provider.get_token()
    assert refreshed.access_token == "new-at"
    assert refreshed.refresh_token == "old-rt"
    assert refreshed.account_id == "acct-after-refresh"
    assert len(calls) == 1
    assert calls[0]["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-rt",
        "client_id": provider.client_id,
    }
    persisted = json.loads(store.read_text())
    assert persisted["access_token"] == "new-at"
    assert persisted["refresh_token"] == "old-rt"


def test_file_provider_raises_when_no_token_exists(tmp_path: Path) -> None:
    provider = FileChatGPTOAuthTokenProvider(path=tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        provider.get_token()


def test_serialize_roundtrip_preserves_fields() -> None:
    token = ChatGPTToken(
        access_token="a",
        refresh_token="b",
        expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        account_id="acct",
        plan_type="plus",
        user_id="u1",
        id_token="id",
    )
    serialized = _serialize_token(token)
    assert serialized["access_token"] == "a"
    assert serialized["expires_at"].endswith("+00:00")
    parsed = json.loads(json.dumps(serialized))
    assert parsed["account_id"] == "acct"


def test_build_authorize_url_includes_pkce_and_state() -> None:
    verifier, challenge = _generate_pkce_pair()
    assert verifier != challenge
    url = _build_authorize_url(
        client_id="app_x",
        redirect_uri="http://localhost:1455/auth/callback",
        state="s1",
        code_challenge=challenge,
    )
    assert "client_id=app_x" in url
    assert "code_challenge_method=S256" in url
    assert "state=s1" in url
    assert "scope=openid+profile+email+offline_access" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback" in url

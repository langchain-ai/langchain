"""Unit tests for `langchain_openai.chatgpt_oauth`."""
# ruff: noqa: S105, S106

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Literal, overload

import httpx
import pytest

from langchain_openai import chatgpt_oauth as oauth_module
from langchain_openai.chatgpt_oauth import (
    CHATGPT_AUTH_CLAIMS_NAMESPACE,
    CHATGPT_TOKEN_URL,
    ChatGPTOAuthRefreshError,
    ChatGPTToken,
    FileChatGPTOAuthTokenProvider,
    _build_authorize_url,
    _CallbackHandler,
    _generate_pkce_pair,
    _serialize_token,
    _token_from_response,
    _wait_for_callback,
    decode_jwt_claims,
    login_chatgpt,
    login_chatgpt_device,
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


def test_pkce_pair_challenge_is_s256_of_verifier() -> None:
    """Regression guard: challenge must be base64url(SHA256(verifier))."""
    verifier, challenge = _generate_pkce_pair()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert challenge == expected


def test_chatgpt_token_repr_does_not_leak_secrets() -> None:
    token = ChatGPTToken(
        access_token="super-secret-at",
        refresh_token="super-secret-rt",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        id_token="super-secret-id",
        account_id="acct-1",
    )
    text = repr(token)
    assert "super-secret-at" not in text
    assert "super-secret-rt" not in text
    assert "super-secret-id" not in text
    assert "acct-1" in text


def test_chatgpt_token_rejects_empty_or_naive_fields() -> None:
    with pytest.raises(ValueError, match="access_token"):
        ChatGPTToken(
            access_token="",
            refresh_token="rt",
            expires_at=datetime.now(timezone.utc),
        )
    with pytest.raises(ValueError, match="refresh_token"):
        ChatGPTToken(
            access_token="at",
            refresh_token="",
            expires_at=datetime.now(timezone.utc),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        ChatGPTToken(
            access_token="at",
            refresh_token="rt",
            expires_at=datetime(2030, 1, 1),  # noqa: DTZ001
        )


def test_token_from_response_raises_on_missing_expires_in() -> None:
    with pytest.raises(ChatGPTOAuthRefreshError, match="expires_in"):
        _token_from_response(
            {"access_token": "a", "refresh_token": "b"},
            fallback_refresh_token=None,
        )


def test_token_from_response_raises_on_missing_refresh_token() -> None:
    with pytest.raises(ChatGPTOAuthRefreshError, match="refresh_token"):
        _token_from_response(
            {"access_token": "a", "expires_in": 3600},
            fallback_refresh_token=None,
        )


def test_token_from_response_raises_on_missing_access_token() -> None:
    with pytest.raises(ChatGPTOAuthRefreshError, match="access_token"):
        _token_from_response(
            {"expires_in": 3600, "refresh_token": "rt"},
            fallback_refresh_token=None,
        )


def test_corrupt_token_store_raises_actionable_error(tmp_path: Path) -> None:
    store = tmp_path / "auth.json"
    store.write_text("{not valid json")
    provider = FileChatGPTOAuthTokenProvider(path=store)
    with pytest.raises(RuntimeError, match="not valid JSON"):
        provider.get_token()


def test_missing_expires_at_in_store_raises_actionable_error(tmp_path: Path) -> None:
    store = tmp_path / "auth.json"
    store.write_text(json.dumps({"access_token": "at", "refresh_token": "rt"}))
    provider = FileChatGPTOAuthTokenProvider(path=store)
    with pytest.raises(RuntimeError, match="missing required"):
        provider.get_token()


def test_invalid_grant_refresh_raises_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "auth.json"
    provider = FileChatGPTOAuthTokenProvider(path=store)
    provider.save(
        ChatGPTToken(
            access_token="old-at",
            refresh_token="old-rt",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
    )

    def _fake_post(*_: Any, **__: Any) -> dict[str, Any]:
        msg = "ChatGPT refresh token is no longer valid (`invalid_grant`)."
        raise ChatGPTOAuthRefreshError(msg)

    monkeypatch.setattr("langchain_openai.chatgpt_oauth._post_form", _fake_post)
    with pytest.raises(ChatGPTOAuthRefreshError, match="invalid_grant"):
        provider.get_token()
    # The on-disk token must be preserved so a follow-up `login_chatgpt()`
    # is the only thing needed.
    persisted = json.loads(store.read_text())
    assert persisted["refresh_token"] == "old-rt"
    assert persisted["access_token"] == "old-at"


def test_refresh_failure_preserves_stored_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "auth.json"
    provider = FileChatGPTOAuthTokenProvider(path=store)
    provider.save(
        ChatGPTToken(
            access_token="keep-at",
            refresh_token="keep-rt",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
    )

    def _explode(*_: Any, **__: Any) -> dict[str, Any]:
        msg = "transient network failure"
        raise RuntimeError(msg)

    monkeypatch.setattr("langchain_openai.chatgpt_oauth._post_form", _explode)
    with pytest.raises(RuntimeError, match="transient"):
        provider.get_token()
    persisted = json.loads(store.read_text())
    assert persisted["refresh_token"] == "keep-rt"
    assert persisted["access_token"] == "keep-at"


def test_aget_token_refreshes_when_expired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "auth.json"
    provider = FileChatGPTOAuthTokenProvider(path=store)
    provider.save(
        ChatGPTToken(
            access_token="old-at",
            refresh_token="old-rt",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
    )

    def _fake_post(_url: str, _data: dict[str, str], **_: Any) -> dict[str, Any]:
        return {"access_token": "new-at", "expires_in": 3600}

    monkeypatch.setattr("langchain_openai.chatgpt_oauth._post_form", _fake_post)
    import asyncio

    refreshed = asyncio.run(provider.aget_token())
    assert refreshed.access_token == "new-at"
    assert refreshed.refresh_token == "old-rt"
    persisted = json.loads(store.read_text())
    assert persisted["access_token"] == "new-at"


def test_aget_access_token_returns_access_string(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "auth.json"
    provider = FileChatGPTOAuthTokenProvider(path=store)
    provider.save(
        ChatGPTToken(
            access_token="at-x",
            refresh_token="rt",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    )

    def _explode(*_: Any, **__: Any) -> dict[str, Any]:
        msg = "should not refresh"
        raise AssertionError(msg)

    monkeypatch.setattr("langchain_openai.chatgpt_oauth._post_form", _explode)
    import asyncio

    assert asyncio.run(provider.aget_access_token()) == "at-x"


def test_token_is_expired_uses_skew_with_frozen_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = datetime(2030, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:  # type: ignore[override]
            return frozen if tz is None else frozen.astimezone(tz)

    monkeypatch.setattr("langchain_openai.chatgpt_oauth.datetime", _FrozenDatetime)
    token = ChatGPTToken(
        access_token="x",
        refresh_token="y",
        expires_at=frozen + timedelta(minutes=1),
    )
    assert token.is_expired(skew=timedelta(minutes=5)) is True
    assert token.is_expired(skew=timedelta(seconds=0)) is False


def _make_response(status_code: int, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )


def test_raise_for_oauth_response_detects_invalid_grant() -> None:
    from langchain_openai.chatgpt_oauth import _raise_for_oauth_response

    resp = _make_response(
        400, {"error": "invalid_grant", "error_description": "revoked"}
    )
    with pytest.raises(ChatGPTOAuthRefreshError, match="invalid_grant"):
        _raise_for_oauth_response(CHATGPT_TOKEN_URL, resp)


def test_raise_for_oauth_response_passes_through_other_errors() -> None:
    from langchain_openai.chatgpt_oauth import _raise_for_oauth_response

    resp = _make_response(500, {"error": "server_error"})
    with pytest.raises(RuntimeError, match="500"):
        _raise_for_oauth_response(CHATGPT_TOKEN_URL, resp)


def test_login_chatgpt_full_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end happy path using a stubbed callback + token endpoint."""
    posts: list[dict[str, Any]] = []
    pkce_holder: list[tuple[str, str]] = []
    real_pkce = oauth_module._generate_pkce_pair

    def _capturing_pkce() -> tuple[str, str]:
        pair = real_pkce()
        pkce_holder.append(pair)
        return pair

    monkeypatch.setattr(oauth_module, "_generate_pkce_pair", _capturing_pkce)
    # Pre-extract the state the SUT will generate by stubbing `secrets.token_urlsafe`
    # so the test can craft a matching callback.
    state_value = "state-xyz"
    monkeypatch.setattr(
        oauth_module.secrets,
        "token_urlsafe",
        lambda _n=32: state_value,
    )

    def _fake_wait_for_callback(**_: Any) -> dict[str, str]:
        return {"code": "auth-code-1", "state": state_value}

    monkeypatch.setattr(oauth_module, "_wait_for_callback", _fake_wait_for_callback)
    # Prevent any browser launch / URL print noise.
    monkeypatch.setattr(oauth_module.webbrowser, "open", lambda _url: True)

    def _fake_post(url: str, data: dict[str, str], **_: Any) -> dict[str, Any]:
        posts.append({"url": url, "data": data})
        return {
            "access_token": "at-new",
            "refresh_token": "rt-new",
            "expires_in": 3600,
        }

    monkeypatch.setattr(oauth_module, "_post_form", _fake_post)

    store = tmp_path / "auth.json"
    provider = login_chatgpt(store_path=store, open_browser=False)

    assert posts[0]["url"] == CHATGPT_TOKEN_URL
    sent = posts[0]["data"]
    assert sent["grant_type"] == "authorization_code"
    assert sent["code"] == "auth-code-1"
    # The verifier the token endpoint sees must match the one paired with
    # the challenge sent to the authorize endpoint.
    assert sent["code_verifier"] == pkce_holder[0][0]
    persisted = json.loads(store.read_text())
    assert persisted["access_token"] == "at-new"
    assert persisted["refresh_token"] == "rt-new"
    assert provider.path == store


def test_login_chatgpt_raises_on_state_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(oauth_module.secrets, "token_urlsafe", lambda _n=32: "expected")
    monkeypatch.setattr(
        oauth_module,
        "_wait_for_callback",
        lambda **_: {"code": "c", "state": "ATTACKER"},
    )
    monkeypatch.setattr(oauth_module, "_post_form", lambda *_a, **_k: {})
    with pytest.raises(RuntimeError, match="state mismatch"):
        login_chatgpt(store_path=tmp_path / "x.json", open_browser=False)


def test_login_chatgpt_state_check_runs_before_error_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If both state and error are present, state mismatch must win."""
    monkeypatch.setattr(oauth_module.secrets, "token_urlsafe", lambda _n=32: "expected")
    monkeypatch.setattr(
        oauth_module,
        "_wait_for_callback",
        lambda **_: {
            "state": "ATTACKER",
            "error": "access_denied",
            "error_description": "user clicked deny",
        },
    )
    monkeypatch.setattr(oauth_module, "_post_form", lambda *_a, **_k: {})
    with pytest.raises(RuntimeError, match="state mismatch"):
        login_chatgpt(store_path=tmp_path / "x.json", open_browser=False)


def test_login_chatgpt_raises_when_code_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(oauth_module.secrets, "token_urlsafe", lambda _n=32: "s")
    monkeypatch.setattr(
        oauth_module,
        "_wait_for_callback",
        lambda **_: {"state": "s"},
    )
    with pytest.raises(RuntimeError, match="authorization code"):
        login_chatgpt(store_path=tmp_path / "x.json", open_browser=False)


def test_login_chatgpt_skips_browser_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    def _track_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(oauth_module.webbrowser, "open", _track_open)
    monkeypatch.setattr(oauth_module.secrets, "token_urlsafe", lambda _n=32: "s")
    monkeypatch.setattr(
        oauth_module,
        "_wait_for_callback",
        lambda **_: {"code": "c", "state": "s"},
    )
    monkeypatch.setattr(
        oauth_module,
        "_post_form",
        lambda *_a, **_k: {
            "access_token": "a",
            "refresh_token": "r",
            "expires_in": 3600,
        },
    )
    login_chatgpt(store_path=tmp_path / "x.json", open_browser=False)
    assert opened == []


def test_login_chatgpt_device_honors_slow_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posts: list[dict[str, Any]] = []
    sleeps: list[float] = []
    responses: list[dict[str, Any]] = [
        {
            "device_code": "dev",
            "user_code": "user",
            "verification_uri": "https://example.com",
        },
        {"error": "authorization_pending"},
        {"error": "slow_down"},
        {"authorization_code": "auth-code"},
        {
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 3600,
        },
    ]
    response_iter = iter(responses)

    def _fake_post(url: str, data: dict[str, str], **_: Any) -> dict[str, Any]:
        posts.append({"url": url, "data": data})
        return next(response_iter)

    def _track_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(oauth_module, "_post_form", _fake_post)
    monkeypatch.setattr(oauth_module.time, "sleep", _track_sleep)

    login_chatgpt_device(store_path=tmp_path / "x.json", poll_interval=2.0)

    # First sleep at base interval, then bumped by +5 after `slow_down`.
    assert sleeps[0] == pytest.approx(2.0)
    assert sleeps[1] == pytest.approx(7.0)


def test_login_chatgpt_device_raises_on_fatal_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses: list[dict[str, Any]] = [
        {
            "device_code": "d",
            "user_code": "u",
            "verification_uri": "https://example.com",
        },
        {"error": "access_denied"},
    ]
    response_iter = iter(responses)
    monkeypatch.setattr(
        oauth_module, "_post_form", lambda *_a, **_k: next(response_iter)
    )
    monkeypatch.setattr(oauth_module.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="access_denied"):
        login_chatgpt_device(store_path=tmp_path / "x.json")


def test_login_chatgpt_device_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        oauth_module,
        "_post_form",
        lambda url, _d, **_k: (
            {
                "device_code": "d",
                "user_code": "u",
                "verification_uri": "https://example.com",
            }
            if url.endswith("usercode")
            else {"error": "authorization_pending"}
        ),
    )
    monkeypatch.setattr(oauth_module.time, "sleep", lambda _s: None)
    # Force the monotonic clock to immediately blow past the deadline.
    times = iter([0.0, 0.0, 9999.0])
    monkeypatch.setattr(oauth_module.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError):
        login_chatgpt_device(
            store_path=tmp_path / "x.json", poll_interval=0.0, timeout=1.0
        )


def test_callback_handler_extracts_code_and_state() -> None:
    result = _run_callback_handler(
        path="/auth/callback?code=abc&state=xyz",
    )
    assert result == {"code": "abc", "state": "xyz"}


def test_callback_handler_404s_unrelated_paths() -> None:
    result = _run_callback_handler(path="/favicon.ico")
    assert result is None


def test_callback_handler_extracts_error() -> None:
    result = _run_callback_handler(
        path="/auth/callback?error=access_denied&error_description=nope",
    )
    assert result == {"error": "access_denied", "error_description": "nope"}


def test_callback_handler_success_renders_success_page() -> None:
    result, body = _run_callback_handler(
        path="/auth/callback?code=abc&state=xyz",
        capture_body=True,
    )
    assert result == {"code": "abc", "state": "xyz"}
    # The apostrophe in "You're" is HTML-escaped by `html.escape`.
    assert "You&#x27;re signed in" in body
    assert "ChatGPT sign-in complete" in body
    assert "Sign-in failed" not in body


def test_callback_handler_error_renders_error_page() -> None:
    result, body = _run_callback_handler(
        path="/auth/callback?error=access_denied&error_description=user+declined",
        capture_body=True,
    )
    assert result == {"error": "access_denied", "error_description": "user declined"}
    assert "Sign-in failed" in body
    assert "user declined" in body
    # Provider error code is surfaced for debuggability.
    assert "access_denied" in body
    assert "You're signed in" not in body


def test_callback_handler_error_without_description_surfaces_code() -> None:
    """The provider's `error` code must reach the user when no description."""
    result, body = _run_callback_handler(
        path="/auth/callback?error=invalid_scope",
        capture_body=True,
    )
    assert result == {"error": "invalid_scope"}
    assert "Sign-in failed" in body
    assert "invalid_scope" in body


def test_callback_handler_escapes_html_in_error_description() -> None:
    """Reflected XSS regression: `error_description` must be HTML-escaped."""
    result, body = _run_callback_handler(
        path=(
            "/auth/callback?error=oops&error_description="
            "%3Cscript%3Ealert(1)%3C%2Fscript%3E"
        ),
        capture_body=True,
    )
    assert result == {
        "error": "oops",
        "error_description": "<script>alert(1)</script>",
    }
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body


def test_callback_handler_error_logs_server_side(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operators need a server-side record of provider OAuth failures."""
    with caplog.at_level("ERROR", logger="langchain_openai.chatgpt_oauth"):
        _run_callback_handler(
            path="/auth/callback?error=access_denied&error_description=nope",
        )
    assert any(
        "access_denied" in rec.message and rec.levelname == "ERROR"
        for rec in caplog.records
    )


def test_wait_for_callback_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the loop to never satisfy the deadline.
    times = iter([0.0, 0.0, 9999.0])
    monkeypatch.setattr(oauth_module.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError):
        _wait_for_callback(
            host="127.0.0.1",
            port=0,
            callback_path="/auth/callback",
            timeout=1.0,
        )


@overload
def _run_callback_handler(
    *, path: str, capture_body: Literal[False] = False
) -> dict[str, str] | None: ...


@overload
def _run_callback_handler(
    *, path: str, capture_body: Literal[True]
) -> tuple[dict[str, str] | None, str]: ...


def _run_callback_handler(
    *, path: str, capture_body: bool = False
) -> dict[str, str] | None | tuple[dict[str, str] | None, str]:
    """Drive a real `_CallbackHandler` against a localhost request.

    Returns the populated `server_result` if the callback was matched, or
    `None` if the handler 404'd. When `capture_body=True`, returns a
    `(result, body)` tuple where `body` is the decoded response body.
    """
    import http.server

    class _BoundCallbackHandler(_CallbackHandler):
        server_result: dict[str, str] = {}

    _BoundCallbackHandler.callback_path = "/auth/callback"
    server = http.server.HTTPServer(("127.0.0.1", 0), _BoundCallbackHandler)
    port = server.server_address[1]
    captured_status: list[int] = []
    captured_body = ""

    def _serve() -> None:
        server.handle_request()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}{path}"
        with urllib.request.urlopen(url, timeout=2.0) as resp:  # noqa: S310
            captured_status.append(resp.status)
            captured_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        captured_status.append(exc.code)
    finally:
        thread.join(timeout=2.0)
        server.server_close()
    result: dict[str, str] | None
    result = (
        None
        if captured_status and captured_status[0] == 404
        else dict(_BoundCallbackHandler.server_result)
    )
    if capture_body:
        return result, captured_body
    return result


def test_file_lock_logs_warning_when_fcntl_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Simulate Windows by making `import fcntl` fail inside `_file_lock`."""
    import builtins

    real_import = builtins.__import__

    def _no_fcntl(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "fcntl":
            msg = "simulated"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_fcntl)
    target = tmp_path / "auth.json"
    with (
        caplog.at_level("WARNING", logger="langchain_openai.chatgpt_oauth"),
        oauth_module._file_lock(target),
    ):
        pass
    assert any("fcntl is unavailable" in rec.message for rec in caplog.records)

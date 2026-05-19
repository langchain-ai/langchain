"""ChatGPT OAuth helpers for `ChatOpenAICodex`.

Implements OAuth 2.0 Authorization Code Flow with PKCE against the OpenAI
auth endpoints used by Codex/ChatGPT subscription auth, plus a small file-backed
token store and refresh logic.

These helpers exist to keep login and token management *separate* from model
invocation. `ChatOpenAICodex` only consumes a `ChatGPTOAuthTokenProvider`.

!!! warning
    This is provider-specific subscription auth and is independent from the
    standard OpenAI API-key flow used by `ChatOpenAI`. Refresh-token rotation
    against `~/.codex/auth.json` can break Codex CLI / VS Code sessions, so
    the default store lives at `~/.langchain/chatgpt-auth.json`.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


CHATGPT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CHATGPT_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CHATGPT_TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105
CHATGPT_DEVICE_CODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
CHATGPT_DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"  # noqa: S105
CHATGPT_DEVICE_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"
CHATGPT_AUTH_CLAIMS_NAMESPACE = "https://api.openai.com/auth"
DEFAULT_REDIRECT_HOST = "localhost"
DEFAULT_REDIRECT_PORT = 1455
DEFAULT_REDIRECT_PATH = "/auth/callback"
DEFAULT_SCOPE = "openid profile email offline_access"
DEFAULT_REFRESH_SKEW = timedelta(minutes=5)
DEFAULT_STORE_PATH = Path.home() / ".langchain" / "chatgpt-auth.json"


@dataclass
class ChatGPTToken:
    """A ChatGPT OAuth token bundle.

    `expires_at` is timezone-aware UTC. Optional fields are populated when
    decodable from the `id_token` JWT.
    """

    access_token: str
    refresh_token: str
    expires_at: datetime
    account_id: str | None = None
    plan_type: str | None = None
    user_id: str | None = None
    id_token: str | None = None

    def is_expired(self, *, skew: timedelta = DEFAULT_REFRESH_SKEW) -> bool:
        """Return True if the token is past (or within `skew` of) expiry."""
        return datetime.now(timezone.utc) >= (self.expires_at - skew)


@runtime_checkable
class ChatGPTOAuthTokenProvider(Protocol):
    """Refresh-aware token source consumed by `ChatOpenAICodex`."""

    def get_token(self) -> ChatGPTToken:
        """Return a current token, refreshing if necessary."""
        ...

    async def aget_token(self) -> ChatGPTToken:
        """Async variant of `get_token`."""
        ...

    def get_access_token(self) -> str:
        """Return only the access token string (sync callable for SDKs)."""
        ...


def _b64url_decode_segment(segment: str) -> bytes:
    """Decode a single base64url JWT segment, handling missing padding."""
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def decode_jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT's payload without signature verification.

    !!! danger
        This is for *local claim extraction only*. Never use the returned
        claims for security or authorization decisions.

    Args:
        token: A JWT (`header.payload.signature`).

    Returns:
        Decoded payload as a dict. Returns an empty dict if the token is
        malformed.
    """
    if not token or token.count(".") < 2:
        return {}
    try:
        _, payload, _ = token.split(".", 2)
        return json.loads(_b64url_decode_segment(payload))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _extract_chatgpt_claims(id_token: str | None) -> dict[str, str | None]:
    """Pull the ChatGPT account/plan/user IDs out of an ID-token JWT."""
    out: dict[str, str | None] = {
        "account_id": None,
        "plan_type": None,
        "user_id": None,
    }
    if not id_token:
        return out
    claims = decode_jwt_claims(id_token)
    auth = claims.get(CHATGPT_AUTH_CLAIMS_NAMESPACE) or {}
    if isinstance(auth, dict):
        out["account_id"] = auth.get("chatgpt_account_id")
        out["plan_type"] = auth.get("chatgpt_plan_type")
        out["user_id"] = auth.get("chatgpt_user_id")
    return out


def _expires_at_from_response(payload: dict[str, Any]) -> datetime:
    expires_in = int(payload.get("expires_in", 0) or 0)
    return datetime.now(timezone.utc) + timedelta(seconds=expires_in)


def _token_from_response(
    payload: dict[str, Any],
    *,
    fallback_refresh_token: str | None = None,
) -> ChatGPTToken:
    """Build a `ChatGPTToken` from an OAuth token-endpoint response."""
    id_token = payload.get("id_token")
    claims = _extract_chatgpt_claims(id_token)
    refresh_token = payload.get("refresh_token") or fallback_refresh_token or ""
    return ChatGPTToken(
        access_token=payload["access_token"],
        refresh_token=refresh_token,
        expires_at=_expires_at_from_response(payload),
        account_id=claims["account_id"],
        plan_type=claims["plan_type"],
        user_id=claims["user_id"],
        id_token=id_token,
    )


def _serialize_token(token: ChatGPTToken) -> dict[str, Any]:
    data = asdict(token)
    data["expires_at"] = token.expires_at.astimezone(timezone.utc).isoformat()
    return data


def _deserialize_token(data: dict[str, Any]) -> ChatGPTToken:
    expires_at_raw = data.get("expires_at")
    if isinstance(expires_at_raw, str):
        expires_at = datetime.fromisoformat(expires_at_raw)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    elif isinstance(expires_at_raw, (int, float)):
        expires_at = datetime.fromtimestamp(expires_at_raw, tz=timezone.utc)
    else:
        msg = "Stored token is missing `expires_at`."
        raise ValueError(msg)
    return ChatGPTToken(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        expires_at=expires_at,
        account_id=data.get("account_id"),
        plan_type=data.get("plan_type"),
        user_id=data.get("user_id"),
        id_token=data.get("id_token"),
    )


def _atomic_write_private_json(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as JSON to `path` with 0600 perms (where supported)."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError, NotImplementedError):
        os.chmod(parent, 0o700)  # noqa: PTH101
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, indent=2, sort_keys=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    tmp.replace(path)
    with contextlib.suppress(OSError, NotImplementedError):
        os.chmod(path, 0o600)  # noqa: PTH101


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Best-effort cross-platform file lock around refresh + write."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        os.close(fd)


def _redact(value: str | None) -> str:
    if not value:
        return "<empty>"
    return f"<redacted len={len(value)}>"


def _post_form(
    url: str,
    data: dict[str, str],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST a form payload and return the parsed JSON body."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            url,
            data=data,
            headers={"Accept": "application/json"},
        )
    if resp.status_code >= 400:
        msg = (
            f"OAuth request to {url} failed with status {resp.status_code}: "
            f"{resp.text[:500]}"
        )
        raise RuntimeError(msg)
    return resp.json()


async def _apost_form(
    url: str,
    data: dict[str, str],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            data=data,
            headers={"Accept": "application/json"},
        )
    if resp.status_code >= 400:
        msg = (
            f"OAuth request to {url} failed with status {resp.status_code}: "
            f"{resp.text[:500]}"
        )
        raise RuntimeError(msg)
    return resp.json()


@dataclass
class FileChatGPTOAuthTokenProvider:
    """File-backed `ChatGPTOAuthTokenProvider`.

    Stores tokens at `path` (default: `~/.langchain/chatgpt-auth.json`) with
    private permissions and refreshes them on read when they are within
    `refresh_skew` of expiry. Refresh token rotation is preserved across
    writes: if the OAuth response omits `refresh_token`, the existing one is
    reused.

    !!! warning
        The default path is intentionally distinct from `~/.codex/auth.json`
        so that refresh-token rotation here does not invalidate Codex CLI /
        VS Code sessions.
    """

    path: Path = field(default_factory=lambda: DEFAULT_STORE_PATH)
    client_id: str = CHATGPT_CLIENT_ID
    token_url: str = CHATGPT_TOKEN_URL
    refresh_skew: timedelta = DEFAULT_REFRESH_SKEW
    timeout: float = 30.0
    _cached: ChatGPTToken | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @classmethod
    def from_default_store(cls) -> FileChatGPTOAuthTokenProvider:
        """Construct a provider rooted at the default store path."""
        return cls()

    def _read_from_disk(self) -> ChatGPTToken | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load ChatGPT token store at %s: %s", self.path, exc
            )
            return None
        try:
            return _deserialize_token(data)
        except (KeyError, ValueError) as exc:
            logger.warning("ChatGPT token store at %s is invalid: %s", self.path, exc)
            return None

    def _write_to_disk(self, token: ChatGPTToken) -> None:
        _atomic_write_private_json(self.path, _serialize_token(token))

    def save(self, token: ChatGPTToken) -> None:
        """Persist `token` to disk and cache it in memory."""
        with self._lock, _file_lock(self.path):
            self._write_to_disk(token)
            self._cached = token

    def _build_refresh_payload(self, refresh_token: str) -> dict[str, str]:
        return {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }

    def _apply_refresh_response(
        self, response: dict[str, Any], previous_refresh: str
    ) -> ChatGPTToken:
        token = _token_from_response(response, fallback_refresh_token=previous_refresh)
        self._write_to_disk(token)
        self._cached = token
        return token

    def _refresh_sync(self, existing: ChatGPTToken) -> ChatGPTToken:
        logger.debug(
            "Refreshing ChatGPT access token (refresh_token=%s).",
            _redact(existing.refresh_token),
        )
        response = _post_form(
            self.token_url,
            self._build_refresh_payload(existing.refresh_token),
            timeout=self.timeout,
        )
        return self._apply_refresh_response(response, existing.refresh_token)

    async def _refresh_async(self, existing: ChatGPTToken) -> ChatGPTToken:
        logger.debug(
            "Refreshing ChatGPT access token (refresh_token=%s).",
            _redact(existing.refresh_token),
        )
        response = await _apost_form(
            self.token_url,
            self._build_refresh_payload(existing.refresh_token),
            timeout=self.timeout,
        )
        return self._apply_refresh_response(response, existing.refresh_token)

    def _load_existing(self) -> ChatGPTToken:
        existing = self._cached or self._read_from_disk()
        if existing is None:
            msg = (
                f"No ChatGPT OAuth token found at {self.path}. Run "
                "`langchain_openai.chatgpt_oauth.login_chatgpt()` first."
            )
            raise FileNotFoundError(msg)
        return existing

    def get_token(self) -> ChatGPTToken:
        """Return a fresh token, refreshing on disk if needed."""
        with self._lock, _file_lock(self.path):
            existing = self._load_existing()
            if not existing.is_expired(skew=self.refresh_skew):
                self._cached = existing
                return existing
            return self._refresh_sync(existing)

    async def aget_token(self) -> ChatGPTToken:
        """Async variant of `get_token`."""
        existing = self._load_existing()
        if not existing.is_expired(skew=self.refresh_skew):
            self._cached = existing
            return existing
        return await self._refresh_async(existing)

    def get_access_token(self) -> str:
        """Return only the access-token string."""
        return self.get_token().access_token


def _generate_pkce_pair() -> tuple[str, str]:
    """Return a `(code_verifier, code_challenge)` pair using S256."""
    verifier = (
        base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    )
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    scope: str = DEFAULT_SCOPE,
    extra_params: dict[str, str] | None = None,
) -> str:
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    if extra_params:
        params.update(extra_params)
    return f"{CHATGPT_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    server_result: dict[str, str] = {}
    callback_path: str = DEFAULT_REDIRECT_PATH

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != self.callback_path:
            self.send_response(404)
            self.end_headers()
            return
        query = urllib.parse.parse_qs(parsed.query)
        for key in ("code", "state", "error", "error_description"):
            value = query.get(key)
            if value:
                self.server_result[key] = value[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        body = (
            "<html><body><h2>Sign-in complete</h2>"
            "<p>You may close this window.</p></body></html>"
        )
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Don't leak callback URLs (which contain auth codes) into stderr.
        return


def _wait_for_callback(
    *,
    host: str,
    port: int,
    callback_path: str,
    timeout: float,
) -> dict[str, str]:
    class _BoundCallbackHandler(_CallbackHandler):
        server_result: dict[str, str] = {}

    _BoundCallbackHandler.callback_path = callback_path
    server = http.server.HTTPServer((host, port), _BoundCallbackHandler)
    server.timeout = 1.0
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            server.handle_request()
            if _BoundCallbackHandler.server_result.get(
                "code"
            ) or _BoundCallbackHandler.server_result.get("error"):
                return dict(_BoundCallbackHandler.server_result)
    finally:
        server.server_close()
    msg = f"Timed out waiting for ChatGPT OAuth callback on http://{host}:{port}"
    raise TimeoutError(msg)


def login_chatgpt(
    *,
    store_path: Path | None = None,
    client_id: str = CHATGPT_CLIENT_ID,
    host: str = DEFAULT_REDIRECT_HOST,
    port: int = DEFAULT_REDIRECT_PORT,
    callback_path: str = DEFAULT_REDIRECT_PATH,
    scope: str = DEFAULT_SCOPE,
    open_browser: bool = True,
    timeout: float = 300.0,
) -> FileChatGPTOAuthTokenProvider:
    """Run the ChatGPT OAuth 2.0 Authorization Code Flow with PKCE.

    Starts a localhost callback server, opens a browser to the OpenAI
    authorize endpoint, exchanges the returned code for tokens, and
    persists them via `FileChatGPTOAuthTokenProvider`.

    Args:
        store_path: Where to persist the token. Defaults to
            `~/.langchain/chatgpt-auth.json`.
        client_id: OAuth client ID (defaults to Codex/ChatGPT client).
        host: Local callback host.
        port: Local callback port.
        callback_path: Local callback path.
        scope: OAuth scope string.
        open_browser: Whether to launch the system browser.
        timeout: Seconds to wait for the callback.

    Returns:
        A `FileChatGPTOAuthTokenProvider` ready for use by
            `ChatOpenAICodex`.
    """
    redirect_uri = f"http://{host}:{port}{callback_path}"
    state = secrets.token_urlsafe(32)
    verifier, challenge = _generate_pkce_pair()
    authorize_url = _build_authorize_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=challenge,
        scope=scope,
    )

    logger.info("Opening ChatGPT sign-in flow at %s", CHATGPT_AUTHORIZE_URL)
    if open_browser:
        with contextlib.suppress(webbrowser.Error):
            webbrowser.open(authorize_url)

    result = _wait_for_callback(
        host=host, port=port, callback_path=callback_path, timeout=timeout
    )
    if "error" in result:
        description = result.get("error_description", "")
        msg = f"ChatGPT OAuth callback returned error: {result['error']} {description}"
        raise RuntimeError(msg)
    if result.get("state") != state:
        msg = "ChatGPT OAuth callback state mismatch."
        raise RuntimeError(msg)
    code = result.get("code")
    if not code:
        msg = "ChatGPT OAuth callback did not include an authorization code."
        raise RuntimeError(msg)

    response = _post_form(
        CHATGPT_TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        },
    )
    token = _token_from_response(response)
    provider = FileChatGPTOAuthTokenProvider(
        path=store_path or DEFAULT_STORE_PATH, client_id=client_id
    )
    provider.save(token)
    return provider


def login_chatgpt_device(
    *,
    store_path: Path | None = None,
    client_id: str = CHATGPT_CLIENT_ID,
    poll_interval: float = 5.0,
    timeout: float = 600.0,
) -> FileChatGPTOAuthTokenProvider:
    """Run the ChatGPT device-code OAuth flow.

    This is the headless fallback for environments without a browser. The
    function prints a verification URL and user code, polls for completion,
    then exchanges the resulting code via the OAuth token endpoint using
    `CHATGPT_DEVICE_REDIRECT_URI`.

    Args:
        store_path: Where to persist the token.
        client_id: OAuth client ID.
        poll_interval: Seconds between polls.
        timeout: Total seconds to wait.

    Returns:
        A configured `FileChatGPTOAuthTokenProvider`.
    """
    _verifier, challenge = _generate_pkce_pair()
    start = _post_form(
        CHATGPT_DEVICE_CODE_URL,
        {
            "client_id": client_id,
            "scope": DEFAULT_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
    )
    device_code = start.get("device_code")
    user_code = start.get("user_code")
    verification_uri = start.get("verification_uri") or start.get(
        "verification_uri_complete"
    )
    if not (device_code and user_code and verification_uri):
        msg = "ChatGPT device-code response missing required fields."
        raise RuntimeError(msg)
    logger.info(
        "Open %s in a browser and enter user code: %s", verification_uri, user_code
    )

    deadline = time.monotonic() + timeout
    authorization_code: str | None = None
    while time.monotonic() < deadline:
        poll = _post_form(
            CHATGPT_DEVICE_TOKEN_URL,
            {"client_id": client_id, "device_code": device_code},
        )
        if poll.get("authorization_code"):
            authorization_code = poll["authorization_code"]
            break
        if poll.get("error") and poll["error"] not in {
            "authorization_pending",
            "slow_down",
        }:
            msg = f"Device authorization failed: {poll['error']}"
            raise RuntimeError(msg)
        time.sleep(poll_interval)
    if not authorization_code:
        msg = "Timed out waiting for ChatGPT device authorization."
        raise TimeoutError(msg)

    response = _post_form(
        CHATGPT_TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": CHATGPT_DEVICE_REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": _verifier,
        },
    )
    token = _token_from_response(response)
    provider = FileChatGPTOAuthTokenProvider(
        path=store_path or DEFAULT_STORE_PATH, client_id=client_id
    )
    provider.save(token)
    return provider


__all__ = [
    "CHATGPT_AUTHORIZE_URL",
    "CHATGPT_CLIENT_ID",
    "CHATGPT_TOKEN_URL",
    "ChatGPTOAuthTokenProvider",
    "ChatGPTToken",
    "FileChatGPTOAuthTokenProvider",
    "decode_jwt_claims",
    "login_chatgpt",
    "login_chatgpt_device",
]

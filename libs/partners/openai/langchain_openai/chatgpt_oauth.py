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

import asyncio
import base64
import contextlib
import hashlib
import html
import http.server
import ipaddress
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

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


@dataclass(frozen=True)
class ChatGPTToken:
    """A ChatGPT OAuth token bundle.

    `expires_at` is timezone-aware. The JWT-derived optionals (`account_id`,
    `plan_type`, `user_id`) are populated when decodable from the `id_token`;
    `id_token` itself is the raw token, not derived from it. Secret-bearing
    fields (`access_token`, `refresh_token`, `id_token`) are excluded from the
    default `repr` so the token does not leak into logs or tracebacks.

    Instances are frozen: the constructor invariants below hold for the life of
    the object, which matters because providers cache and share a single token
    and replace it wholesale on refresh rather than mutating fields in place.
    """

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: datetime
    account_id: str | None = None
    plan_type: str | None = None
    user_id: str | None = None
    id_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate non-empty secrets and timezone-aware `expires_at`."""
        if not self.access_token:
            msg = "`access_token` must be a non-empty string."
            raise ValueError(msg)
        if not self.refresh_token:
            msg = "`refresh_token` must be a non-empty string."
            raise ValueError(msg)
        if self.expires_at.tzinfo is None:
            msg = "`expires_at` must be timezone-aware (UTC)."
            raise ValueError(msg)

    def is_expired(self, *, skew: timedelta = DEFAULT_REFRESH_SKEW) -> bool:
        """Return `True` if the token is past (or within `skew` of) expiry."""
        return datetime.now(timezone.utc) >= (self.expires_at - skew)


class ChatGPTOAuthRefreshError(RuntimeError):
    """Raised when a refresh-token grant fails irrecoverably.

    Typically signals that the stored refresh token has been revoked or has
    expired; the caller should re-run `login_chatgpt()` (or the device-code
    equivalent) to obtain a new bundle.
    """


@runtime_checkable
class ChatGPTOAuthTokenProvider(Protocol):
    """Refresh-aware token source consumed by `ChatOpenAICodex`."""

    def get_token(self) -> ChatGPTToken:
        """Return a current token, refreshing if necessary."""
        ...

    async def aget_token(self) -> ChatGPTToken:
        """Async variant of `get_token`.

        Implementations must offer the same locking and refresh guarantees
        as `get_token`: concurrent callers must not race on token storage.
        """
        ...

    def get_access_token(self) -> str:
        """Return only the access token string (sync callable for SDKs)."""
        ...

    async def aget_access_token(self) -> str:
        """Return only the access token string (async callable for SDKs)."""
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
    if out["account_id"] is None:
        # A present-but-unparseable id_token (or one missing the namespaced
        # auth claim) silently drops the `ChatGPT-Account-Id` header, which
        # surfaces later as an opaque backend rejection. Leave a breadcrumb.
        logger.debug(
            "No `chatgpt_account_id` claim extracted from the ChatGPT "
            "id_token; the `ChatGPT-Account-Id` header will be omitted."
        )
    return out


def _expires_at_from_response(payload: dict[str, Any]) -> datetime:
    raw = payload.get("expires_in")
    try:
        expires_in = int(raw) if raw is not None else 0
    except (TypeError, ValueError) as exc:
        msg = f"OAuth token response had invalid `expires_in`: {raw!r}"
        raise ChatGPTOAuthRefreshError(msg) from exc
    if expires_in <= 0:
        msg = (
            "OAuth token response had missing or non-positive `expires_in`; "
            "refusing to store an immediately-expired token."
        )
        raise ChatGPTOAuthRefreshError(msg)
    return datetime.now(timezone.utc) + timedelta(seconds=expires_in)


def _token_from_response(
    payload: dict[str, Any],
    *,
    fallback_refresh_token: str | None = None,
) -> ChatGPTToken:
    """Build a `ChatGPTToken` from an OAuth token-endpoint response."""
    if not payload.get("access_token"):
        msg = "OAuth token response did not include an `access_token`."
        raise ChatGPTOAuthRefreshError(msg)
    id_token = payload.get("id_token")
    claims = _extract_chatgpt_claims(id_token)
    refresh_token = payload.get("refresh_token") or fallback_refresh_token
    if not refresh_token:
        msg = (
            "OAuth token response did not include a `refresh_token` and no "
            "prior refresh token was available; re-run `login_chatgpt()`."
        )
        raise ChatGPTOAuthRefreshError(msg)
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
    return {
        "access_token": token.access_token,
        "refresh_token": token.refresh_token,
        "expires_at": token.expires_at.astimezone(timezone.utc).isoformat(),
        "account_id": token.account_id,
        "plan_type": token.plan_type,
        "user_id": token.user_id,
        "id_token": token.id_token,
    }


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


def _chmod_warn(path: Path, mode: int) -> None:
    """Best-effort `chmod` that logs (but does not raise) on failure.

    On filesystems without POSIX perms (Windows, some FUSE/SMB mounts) the
    file may end up world-readable. Logging surfaces that to operators so
    they don't silently trust the "private perms" claim of the caller.
    """
    try:
        os.chmod(path, mode)  # noqa: PTH101
    except (OSError, NotImplementedError) as exc:
        logger.warning(
            "Failed to set permissions %o on %s: %s — token store may not "
            "have private permissions on this filesystem.",
            mode,
            path,
            exc,
        )


def _atomic_write_private_json(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as JSON to `path` with 0600 perms (where supported)."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    _chmod_warn(parent, 0o700)
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
    _chmod_warn(path, 0o600)


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Best-effort cross-platform file lock around refresh + write.

    On POSIX this acquires an exclusive `fcntl.flock` on a sibling
    `.lock` file. On Windows (or any platform where `fcntl` is
    unavailable) the lock degrades to a no-op and a warning is logged so
    callers know that cross-process safety is best-effort.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        try:
            import fcntl
        except ImportError:
            logger.warning(
                "fcntl is unavailable on this platform; ChatGPT token store "
                "at %s is not protected against cross-process races.",
                path,
            )
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                locked = True
            except OSError as exc:
                logger.warning(
                    "fcntl.flock failed on %s: %s — token store is not "
                    "protected against cross-process races.",
                    lock_path,
                    exc,
                )
        yield
    finally:
        if locked:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            except (ImportError, OSError) as exc:
                logger.warning("Failed to release file lock on %s: %s", lock_path, exc)
        os.close(fd)


def _redact(value: str | None) -> str:
    if not value:
        return "<empty>"
    return f"<redacted len={len(value)}>"


def _parse_oauth_error(resp: httpx.Response) -> tuple[str | None, str]:
    """Return `(error_code, body_excerpt)` from an OAuth error response."""
    try:
        payload = resp.json()
    except (ValueError, json.JSONDecodeError):
        return None, resp.text[:500]
    if isinstance(payload, dict):
        error = payload.get("error")
        description = payload.get("error_description") or ""
        excerpt = f"{error}: {description}".strip(": ") or resp.text[:500]
        return (error if isinstance(error, str) else None), excerpt
    return None, resp.text[:500]


def _raise_for_oauth_response(url: str, resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    error_code, excerpt = _parse_oauth_error(resp)
    if error_code == "invalid_grant":
        msg = (
            "ChatGPT refresh token is no longer valid (`invalid_grant`). "
            "Re-run `login_chatgpt()` to obtain a new token."
        )
        raise ChatGPTOAuthRefreshError(msg)
    msg = f"OAuth request to {url} failed with status {resp.status_code}: {excerpt}"
    raise RuntimeError(msg)


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
    _raise_for_oauth_response(url, resp)
    return resp.json()


_DEVICE_POLL_PENDING_ERRORS = frozenset({"authorization_pending", "slow_down"})


def _post_device_poll_form(
    url: str,
    data: dict[str, str],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST a device-code poll and return expected pending error payloads."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            url,
            data=data,
            headers={"Accept": "application/json"},
        )
    if resp.status_code < 400:
        return resp.json()
    error_code, _ = _parse_oauth_error(resp)
    if error_code in _DEVICE_POLL_PENDING_ERRORS:
        return resp.json()
    _raise_for_oauth_response(url, resp)
    return resp.json()


async def _apost_form(
    url: str,
    data: dict[str, str],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST a form payload asynchronously and return the parsed JSON body."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            data=data,
            headers={"Accept": "application/json"},
        )
    _raise_for_oauth_response(url, resp)
    return resp.json()


@dataclass
class FileChatGPTOAuthTokenProvider:
    """File-backed `ChatGPTOAuthTokenProvider`.

    Stores tokens at `path` (defaults to `DEFAULT_STORE_PATH`) with private
    permissions and refreshes them on read when they are within
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
        """Construct a provider with all defaults (path, client ID, etc.).

        Equivalent to `FileChatGPTOAuthTokenProvider()`; the alias exists as
        a discoverable entry point for callers reading the default-path
        contract from the module docstring.
        """
        return cls()

    def _read_from_disk(self) -> ChatGPTToken | None:
        """Return the stored token, or `None` if no store exists.

        Raises `RuntimeError` (rather than returning `None`) if the file
        exists but cannot be parsed — that way the user is not told to
        "re-login" when the actual fix is to repair or remove a corrupt
        store at `self.path`.
        """
        if not self.path.exists():
            return None
        try:
            raw_text = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            msg = (
                f"Failed to read ChatGPT token store at {self.path}: {exc}. "
                "Repair file permissions/encoding or delete the file and "
                "re-run `login_chatgpt()`."
            )
            raise RuntimeError(msg) from exc
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            msg = (
                f"ChatGPT token store at {self.path} is not valid JSON: "
                f"{exc}. Delete the file and re-run `login_chatgpt()`."
            )
            raise RuntimeError(msg) from exc
        try:
            return _deserialize_token(data)
        except (KeyError, ValueError) as exc:
            msg = (
                f"ChatGPT token store at {self.path} is missing required "
                f"fields ({exc}). Delete the file and re-run "
                "`login_chatgpt()`."
            )
            raise RuntimeError(msg) from exc

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

    def _load_existing(self) -> ChatGPTToken:
        existing = self._cached or self._read_from_disk()
        if existing is None:
            msg = (
                f"No ChatGPT OAuth token found at {self.path}. Run "
                "`langchain_openai.chatgpt_oauth.login_chatgpt()` first."
            )
            raise FileNotFoundError(msg)
        return existing

    def _load_existing_before_refresh(self) -> ChatGPTToken:
        existing = self._load_existing()
        if not existing.is_expired(skew=self.refresh_skew):
            return existing
        disk_token = self._read_from_disk()
        if disk_token is not None:
            self._cached = disk_token
            return disk_token
        return existing

    def get_token(self) -> ChatGPTToken:
        """Return a fresh token, refreshing on disk if needed.

        Raises:
            FileNotFoundError: No token store exists at `self.path`; run
                `login_chatgpt()` first.
            ChatGPTOAuthRefreshError: The stored refresh token was rejected
                (e.g. revoked or expired); re-run `login_chatgpt()`.
        """
        with self._lock, _file_lock(self.path):
            existing = self._load_existing_before_refresh()
            if not existing.is_expired(skew=self.refresh_skew):
                self._cached = existing
                return existing
            return self._refresh_sync(existing)

    async def aget_token(self) -> ChatGPTToken:
        """Async variant of `get_token` with the same locking guarantees.

        The thread lock and cross-process file lock are acquired off the
        event loop via `asyncio.to_thread` so concurrent async callers do
        not race on `_cached` or on the on-disk token bundle. The HTTP
        refresh runs synchronously inside that worker thread; this avoids
        nesting event loops while still keeping the cross-process lock
        held for the entire refresh + write window.

        Raises:
            FileNotFoundError: No token store exists at `self.path`; run
                `login_chatgpt()` first.
            ChatGPTOAuthRefreshError: The stored refresh token was rejected
                (e.g. revoked or expired); re-run `login_chatgpt()`.
        """
        return await asyncio.to_thread(self._aget_token_locked_blocking)

    def _aget_token_locked_blocking(self) -> ChatGPTToken:
        with self._lock, _file_lock(self.path):
            existing = self._load_existing_before_refresh()
            if not existing.is_expired(skew=self.refresh_skew):
                self._cached = existing
                return existing
            return self._refresh_sync(existing)

    def get_access_token(self) -> str:
        """Return only the access-token string."""
        return self.get_token().access_token

    async def aget_access_token(self) -> str:
        """Return only the access-token string (async)."""
        token = await self.aget_token()
        return token.access_token


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
            # Surface path mismatches: otherwise a misconfigured
            # `callback_path` looks identical to "still waiting" and only
            # ends in a generic timeout. (Path only — never the query, which
            # carries the auth code.)
            logger.debug(
                "Ignoring callback request for unexpected path %r (expected %r).",
                parsed.path,
                self.callback_path,
            )
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
        error = self.server_result.get("error")
        if error:
            error_description = self.server_result.get("error_description")
            logger.error(
                "ChatGPT OAuth callback returned error %r (%s)",
                error,
                error_description or "no description",
            )
            if error_description:
                description = f"{error_description} (error: {error})"
            else:
                description = (
                    f"ChatGPT returned error '{error}'. Close this tab and "
                    "try `login_chatgpt()` again."
                )
            body = _oauth_error_html(description)
        else:
            body = _oauth_success_html(
                "ChatGPT sign-in complete. You can close this browser tab "
                "and return to your terminal.",
            )
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Don't leak callback URLs (which contain auth codes) into stderr.
        return


def _oauth_success_html(message: str) -> str:
    return _oauth_result_html(
        title="ChatGPT sign-in complete",
        heading="You're signed in",
        message=message,
        status="success",
    )


def _oauth_error_html(message: str) -> str:
    return _oauth_result_html(
        title="ChatGPT sign-in failed",
        heading="Sign-in failed",
        message=message,
        status="error",
    )


def _oauth_result_html(
    *,
    title: str,
    heading: str,
    message: str,
    status: Literal["success", "error"],
) -> str:
    accent = "#137333" if status == "success" else "#b3261e"
    background = "#eef7f0" if status == "success" else "#fceeee"
    mark = "&check;" if status == "success" else "!"
    escaped_title = html.escape(title)
    escaped_heading = html.escape(heading)
    escaped_message = html.escape(message)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escaped_title}</title>"
        "<style>"
        "body{margin:0;min-height:100vh;display:grid;place-items:center;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "background:#f8faf9;color:#1f2328}"
        ".panel{width:min(480px,calc(100vw - 40px));box-sizing:border-box;"
        "padding:32px;border:1px solid #d8dee4;border-radius:8px;"
        "background:#fff;box-shadow:0 18px 45px rgba(31,35,40,.08)}"
        ".mark{width:44px;height:44px;border-radius:50%;display:grid;"
        "place-items:center;margin-bottom:20px;font-weight:700;font-size:22px}"
        "h1{font-size:24px;line-height:1.2;margin:0 0 10px}"
        "p{font-size:15px;line-height:1.5;margin:0;color:#57606a}"
        "@media (prefers-color-scheme: dark){"
        "body{background:#0d1117;color:#e6edf3}"
        ".panel{background:#161b22;border-color:#30363d;"
        "box-shadow:0 18px 45px rgba(0,0,0,.4)}"
        "p{color:#9da7b3}}"
        "</style></head><body>"
        '<main class="panel">'
        f'<div class="mark" style="background:{background};color:{accent}">'
        f"{mark}</div>"
        f"<h1>{escaped_heading}</h1><p>{escaped_message}</p>"
        "</main>"
        "</body></html>"
    )


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
    try:
        server = http.server.HTTPServer((host, port), _BoundCallbackHandler)
    except OSError as exc:
        msg = (
            f"Could not bind ChatGPT OAuth callback server on "
            f"http://{host}:{port}: {exc}. Free the port or pass `port=` "
            "to `login_chatgpt()` with an unused port."
        )
        raise RuntimeError(msg) from exc
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


def _validate_loopback_host(host: str) -> None:
    """Reject non-loopback callback hosts.

    The callback server receives the OAuth authorization `code` in the request
    URL. Binding it to a non-loopback interface (e.g. `0.0.0.0`) would expose
    that code on the local network, so only loopback hosts are permitted —
    RFC 8252 §8.3 expects a loopback redirect for native-app PKCE flows.

    Args:
        host: The callback host passed to `login_chatgpt`.

    Raises:
        ValueError: `host` is not `localhost` or a loopback IP address.
    """
    if host == "localhost":
        return
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Not an IP literal and not `localhost`; can't prove it's loopback.
        is_loopback = False
    if not is_loopback:
        msg = (
            f"`host={host!r}` is not a loopback address. The OAuth callback "
            "server receives the authorization code in the request URL, so it "
            "must bind to a loopback interface (`localhost`, `127.0.0.1`, or "
            "`::1`) to avoid exposing the code on the network."
        )
        raise ValueError(msg)


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

    Starts a loopback callback server, optionally opens a browser to the
    OpenAI authorize endpoint (when `open_browser=True`; the URL is always
    printed as a fallback), exchanges the returned code for tokens, and
    persists them via `FileChatGPTOAuthTokenProvider`.

    Args:
        store_path: Where to persist the token. Defaults to
            `DEFAULT_STORE_PATH`.
        client_id: OAuth client ID (defaults to Codex/ChatGPT client).
        host: Local callback host. Must be a loopback address.
        port: Local callback port.
        callback_path: Local callback path.
        scope: OAuth scope string.
        open_browser: Whether to launch the system browser.
        timeout: Seconds to wait for the callback.

    Returns:
        A `FileChatGPTOAuthTokenProvider` ready for use by
            `ChatOpenAICodex`.

    Raises:
        ValueError: `host` is not a loopback address.
        RuntimeError: The callback server could not bind, the `state` did not
            match (CSRF), the provider returned an OAuth error, or no
            authorization code was returned.
        TimeoutError: No callback was received within `timeout` seconds.

    See Also:
        `login_chatgpt_device`: Headless fallback for environments without a
            browser or the ability to bind a localhost callback port.
    """
    _validate_loopback_host(host)
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

    # Surface the URL prominently so users can complete sign-in manually if
    # the browser launch fails or the environment is headless.
    print(  # noqa: T201
        f"\nChatGPT sign-in: open the following URL in a browser:\n  {authorize_url}\n"
    )
    logger.info("Opening ChatGPT sign-in flow at %s", CHATGPT_AUTHORIZE_URL)
    if open_browser:
        try:
            webbrowser.open(authorize_url)
        except webbrowser.Error as exc:
            logger.warning(
                "Could not launch a browser: %s. Copy the URL above instead.",
                exc,
            )

    result = _wait_for_callback(
        host=host, port=port, callback_path=callback_path, timeout=timeout
    )
    # Validate `state` first: a CSRF mismatch is a security signal and must
    # fail closed before any other branch (including server-reported errors)
    # is considered.
    if result.get("state") != state:
        msg = "ChatGPT OAuth callback state mismatch."
        raise RuntimeError(msg)
    if "error" in result:
        description = result.get("error_description", "")
        msg = f"ChatGPT OAuth callback returned error: {result['error']} {description}"
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
        store_path: Where to persist the token. Defaults to
            `DEFAULT_STORE_PATH`.
        client_id: OAuth client ID (defaults to Codex/ChatGPT client).
        poll_interval: Seconds between polls.
        timeout: Total seconds to wait.

    Returns:
        A configured `FileChatGPTOAuthTokenProvider`.

    Raises:
        RuntimeError: The device-code response was missing required fields, or
            device authorization failed with a terminal error.
        TimeoutError: Authorization was not completed within `timeout` seconds.

    See Also:
        `login_chatgpt`: Browser-based loopback flow preferred when a local
            browser and free callback port are available.
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
    current_interval = poll_interval
    while time.monotonic() < deadline:
        poll = _post_device_poll_form(
            CHATGPT_DEVICE_TOKEN_URL,
            {"client_id": client_id, "device_code": device_code},
        )
        if poll.get("authorization_code"):
            authorization_code = poll["authorization_code"]
            break
        error = poll.get("error")
        if error == "slow_down":
            # RFC 8628 §3.5: bump the poll interval by 5 seconds to comply
            # with server-side rate limiting; otherwise we risk being banned.
            current_interval += 5
        elif error and error != "authorization_pending":
            msg = f"Device authorization failed: {error}"
            raise RuntimeError(msg)
        time.sleep(current_interval)
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
    "ChatGPTOAuthRefreshError",
    "ChatGPTOAuthTokenProvider",
    "ChatGPTToken",
    "FileChatGPTOAuthTokenProvider",
    "decode_jwt_claims",
    "login_chatgpt",
    "login_chatgpt_device",
]

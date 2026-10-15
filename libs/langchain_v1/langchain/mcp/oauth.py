"""OAuth for remote MCP servers that a host application connects on a user's behalf.

`MCPOAuthProvider` extends the MCP SDK's `OAuthClientProvider`. Discovery,
dynamic client registration, PKCE, RFC 9207 issuer checks, and token parsing
all stay in the SDK. This module adds what a hosted application needs on top:

- A login that can pause and resume. `begin` returns an authorization URL and a
    serializable `PendingAuthorization`. `complete` finishes the code exchange
    from that state later, in any process.
- A credential store that persists tokens with their absolute expiry, the
    registered client, and the discovered authorization-server metadata, and
    restores all of it on load. After a restart an expired token is refreshed
    instead of being sent, and the refresh goes to the advertised token
    endpoint.
- One injectable request sender for every request the SDK makes to an
    authorization server, so hosts can apply SSRF guards and response limits.
    Those requests never pass through the MCP resource client, so its default
    headers and hooks never reach discovery, registration, or token endpoints.
- Credential-safe failures. Errors name an HTTP status and at most an RFC 6749
    error code, never a response body, and SDK log records emitted during these
    flows are stripped of tracebacks that embed provider responses.
- Typed events for a host handler, so authorization URLs reach the user without
    passing through model context.

Hosts keep user and session binding, encrypted storage, callback routes, and
cross-replica refresh locking.

!!! warning "This module is in beta"

    The API may change. It relies on internal hooks of the MCP SDK's
    `OAuthClientProvider`, which this module tracks across the SDK's 2.x line.
"""

import contextlib
import logging
import re
import secrets
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias
from urllib.parse import urlencode, urljoin

import httpx2
from mcp.client.auth import OAuthClientProvider, OAuthFlowError, OAuthTokenError
from mcp.client.auth.oauth2 import PKCEParameters
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
    create_oauth_metadata_request,
    handle_auth_metadata_response,
    handle_protected_resource_response,
    validate_authorization_response_iss,
    validate_metadata_issuer,
)
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)
from mcp.shared.inbound import MCP_PROTOCOL_VERSION_HEADER
from mcp.types.version import LATEST_PROTOCOL_VERSION
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from mcp.shared.auth import AuthorizationCodeResult

__all__ = [
    "AuthorizationCompleted",
    "AuthorizationFailed",
    "AuthorizationRequired",
    "AuthorizationStart",
    "InMemoryOAuthCredentialStore",
    "MCPAuthorizationRequiredError",
    "MCPOAuthError",
    "MCPOAuthProvider",
    "MCPOAuthUnavailableError",
    "OAuthCredentialStore",
    "OAuthCredentials",
    "OAuthEvent",
    "OAuthEventHandler",
    "OAuthRequestSender",
    "PendingAuthorization",
    "ReconnectRequired",
]

logger = logging.getLogger(__name__)

_SDK_LOGGER: Final = "mcp.client.auth.oauth2"
"""Logger the SDK's `OAuthClientProvider` writes to."""

_DEFAULT_TIMEOUT_SECONDS: Final = 30.0
_DEFAULT_MAX_RESPONSE_BYTES: Final = 1024 * 1024
_DEFAULT_PENDING_TTL_SECONDS: Final = 600.0
_DEFAULT_EXPIRY_SKEW_SECONDS: Final = 30.0
_CHALLENGES: Final = frozenset({401, 403})
_UNREACHABLE: Final = (httpx2.ConnectError, httpx2.ConnectTimeout)
"""Transport errors raised before a request could reach the server, so safe to retry."""
_TOO_MANY_REQUESTS: Final = 429
_SERVER_ERROR: Final = 500
_TOKEN_SUCCESS: Final = frozenset({200, 201})
_OK: Final = 200

_ERROR_CODE: Final = re.compile(r"[a-z_]{1,64}")
"""Shape of an RFC 6749 §5.2 `error` value, the only response text errors repeat."""


class MCPOAuthError(RuntimeError):
    """An MCP OAuth operation failed.

    Messages never contain tokens, codes, authorization URLs, or provider
    response bodies, so hosts may show them to users or models.
    """


class MCPAuthorizationRequiredError(MCPOAuthError):
    """The user must authorize, or reauthorize, before the server can be used.

    Raised when a request needs a new grant and no in-process callback can
    collect one, and when a refresh is rejected. The authorization URL, when
    one was created, is delivered to the host's event handler instead.
    """


class MCPOAuthUnavailableError(MCPOAuthError):
    """The authorization server could not be reached or failed temporarily.

    Raised for connection failures, HTTP 429, and 5xx responses: cases where
    the request was not processed. Stored credentials are left intact, since
    this is not evidence that the user's grant was revoked. Other transport
    errors, such as a read timeout, propagate unchanged because the server may
    already have rotated a refresh token.
    """


class OAuthCredentials(BaseModel):
    """Everything an `MCPOAuthProvider` persists for one server connection.

    Tokens and client secrets are excluded from `repr`. Hosts that store this
    should encrypt it; `model_dump_json` round-trips through `model_validate_json`.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    tokens: OAuthToken | None = Field(default=None, repr=False)
    """The current grant."""

    expires_at: float | None = None
    """Absolute access-token expiry as a Unix timestamp, or `None` if unknown.

    The SDK's token model only carries a relative `expires_in`, which is
    meaningless once the process that received it is gone.
    """

    client_info: OAuthClientInformationFull | None = Field(default=None, repr=False)
    """The registered OAuth client."""

    oauth_metadata: OAuthMetadata | None = None
    """Discovered authorization-server metadata, including its token endpoint."""

    protected_resource_metadata: ProtectedResourceMetadata | None = None
    """Discovered protected-resource metadata for the MCP server."""

    auth_server_url: str | None = None
    """The authorization server the protected-resource metadata pointed to."""


class OAuthCredentialStore(Protocol):
    """Persistence for one server connection's `OAuthCredentials`.

    The provider loads a snapshot once per instance and saves a complete
    snapshot whenever credentials change, so a store never sees a token
    without its expiry or a refresh without its metadata. Hosts own
    encryption and the identity the snapshot is keyed by.
    """

    async def load(self) -> OAuthCredentials | None:
        """Return the stored credentials, or `None` if there are none."""
        ...

    async def save(self, credentials: OAuthCredentials) -> None:
        """Replace the stored credentials."""
        ...


class InMemoryOAuthCredentialStore:
    """An `OAuthCredentialStore` holding one snapshot in memory.

    Useful for tests, and for hosts that seal the snapshot into their own
    record after an operation completes.
    """

    def __init__(self, credentials: OAuthCredentials | None = None) -> None:
        """Create a store, optionally seeded with credentials.

        Args:
            credentials: Initial snapshot.
        """
        self.credentials = credentials

    async def load(self) -> OAuthCredentials | None:
        """Return the held snapshot."""
        return self.credentials

    async def save(self, credentials: OAuthCredentials) -> None:
        """Replace the held snapshot."""
        self.credentials = credentials


class PendingAuthorization(BaseModel):
    """Serializable state of a login between `begin` and `complete`.

    Contains the PKCE verifier, so hosts should keep it server-side and
    encrypted, bound to the user and browser session that started the login.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    state: str = Field(repr=False)
    """The `state` parameter sent with the authorization request."""

    code_verifier: str = Field(repr=False)
    """The PKCE verifier for the authorization code."""

    redirect_uri: str
    """The redirect URI the authorization request used."""

    scope: str | None = None
    """The scope the authorization request asked for."""

    protocol_version: str | None = None
    """MCP protocol version seen during discovery; decides RFC 8707 `resource`."""

    expires_at: float
    """Unix timestamp after which `complete` rejects this login."""


@dataclass(frozen=True, slots=True)
class AuthorizationStart:
    """Result of `MCPOAuthProvider.begin`."""

    authorization_url: str = field(repr=False)
    """URL to send the user's browser to. Keep it out of model context."""

    pending: PendingAuthorization
    """State to persist until the callback arrives."""


@dataclass(frozen=True, slots=True)
class AuthorizationRequired:
    """A request needs the user to authorize, and a URL is ready for them."""

    server_url: str
    authorization_url: str = field(repr=False)
    pending: PendingAuthorization = field(repr=False)
    type: Literal["authorization_required"] = "authorization_required"


@dataclass(frozen=True, slots=True)
class AuthorizationCompleted:
    """`complete` exchanged the authorization code and stored the grant."""

    server_url: str
    type: Literal["authorization_completed"] = "authorization_completed"


AuthorizationFailureReason: TypeAlias = Literal[
    "expired",
    "invalid_callback",
    "rejected",
    "unavailable",
    "error",
]


@dataclass(frozen=True, slots=True)
class AuthorizationFailed:
    """`complete` could not finish the login."""

    server_url: str
    reason: AuthorizationFailureReason
    type: Literal["authorization_failed"] = "authorization_failed"


@dataclass(frozen=True, slots=True)
class ReconnectRequired:
    """`refresh` was rejected; the user has to connect again."""

    server_url: str
    type: Literal["reconnect_required"] = "reconnect_required"


OAuthEvent: TypeAlias = (
    AuthorizationRequired | AuthorizationCompleted | AuthorizationFailed | ReconnectRequired
)
"""Events an `MCPOAuthProvider` delivers to its host."""

OAuthEventHandler: TypeAlias = Callable[[OAuthEvent], Awaitable[None]]
"""Host callback for `OAuthEvent`s. Runs outside model context."""

OAuthRequestSender: TypeAlias = Callable[[httpx2.Request], Awaitable[httpx2.Response]]
"""Sends one authorization-server request and returns its fully read response."""


_IN_FLOW: ContextVar[bool] = ContextVar("langchain_mcp_oauth_in_flow", default=False)


class _AuthorizationPausedError(Exception):
    """Unwinds the SDK flow once an authorization URL exists."""

    def __init__(self, start: AuthorizationStart) -> None:
        super().__init__("authorization paused")
        self.start = start


class _ScrubSDKRecords(logging.Filter):
    """Strip SDK log records that may embed provider responses.

    The SDK logs flow errors with `logger.exception`, and its token and
    registration errors quote response bodies. While one of this module's
    flows runs, records keep their message but lose their traceback, and the
    record for this module's own pause is dropped.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not _IN_FLOW.get() or not record.exc_info:
            return True
        error = record.exc_info[1]
        if isinstance(error, _AuthorizationPausedError):
            return False
        record.msg = f"{record.getMessage()} ({type(error).__name__})"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        return True


logging.getLogger(_SDK_LOGGER).addFilter(_ScrubSDKRecords())


@contextlib.contextmanager
def _in_flow() -> Iterator[None]:
    token = _IN_FLOW.set(True)
    try:
        yield
    finally:
        _IN_FLOW.reset(token)


def _error_code(response: httpx2.Response) -> str:
    """Return ` (code)` for an RFC 6749 error response, or an empty string."""
    try:
        code = response.json().get("error")
    except (ValueError, AttributeError):
        return ""
    return f" ({code})" if isinstance(code, str) and _ERROR_CODE.fullmatch(code) else ""


async def _send_with_default_client(request: httpx2.Request) -> httpx2.Response:
    """Send an authorization-server request on a fresh client without redirects."""
    async with httpx2.AsyncClient(
        follow_redirects=False, timeout=_DEFAULT_TIMEOUT_SECONDS
    ) as client:
        response = await client.send(request, stream=True)
        try:
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > _DEFAULT_MAX_RESPONSE_BYTES:
                    msg = "OAuth response exceeded the size limit"
                    raise MCPOAuthError(msg)
        finally:
            await response.aclose()
    headers = response.headers.copy()
    headers.pop("content-encoding", None)
    headers.pop("content-length", None)
    return httpx2.Response(
        response.status_code, headers=headers, content=bytes(body), request=request
    )


class _StoreAdapter:
    """Presents an `OAuthCredentialStore` as the SDK's `TokenStorage`."""

    def __init__(self, store: OAuthCredentialStore) -> None:
        self.store = store
        self.provider: MCPOAuthProvider | None = None

    async def get_tokens(self) -> OAuthToken | None:
        credentials = await self.store.load()
        return None if credentials is None else credentials.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        if self.provider is not None:
            await self.provider._save(tokens=tokens)  # noqa: SLF001

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        credentials = await self.store.load()
        return None if credentials is None else credentials.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        del client_info
        if self.provider is not None:
            await self.provider._save()  # noqa: SLF001


class MCPOAuthProvider(OAuthClientProvider):
    """An MCP SDK `OAuthClientProvider` for hosted and long-lived applications.

    Pass it as the `auth` of an MCP HTTP transport. Requests behave like the
    SDK's provider, with three differences: credentials load from and save to
    an `OAuthCredentialStore`, an expired access token is refreshed before it
    is sent, and every authorization-server request goes through
    `request_sender` rather than the resource client.

    When a request needs a new grant, the SDK's `redirect_handler` and
    `callback_handler` run as usual if both are set. Otherwise the provider
    delivers an `AuthorizationRequired` event and raises
    `MCPAuthorizationRequiredError`.

    Hosts that connect users from a web UI call `begin` and `complete`
    directly, and `refresh` to rotate tokens under their own locking.

    Example:
        ```python
        provider = MCPOAuthProvider(
            server_url="https://mcp.example.com/mcp",
            client_metadata=OAuthClientMetadata(
                client_name="Example",
                redirect_uris=["https://app.example.com/oauth/callback"],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="none",
            ),
            store=store,
        )
        start = await provider.begin()
        # Persist start.pending, redirect the browser to start.authorization_url,
        # then from the callback route, possibly in another process:
        await provider.complete(pending, code=code, state=state, iss=iss)
        ```
    """

    def __init__(
        self,
        server_url: str,
        client_metadata: OAuthClientMetadata,
        store: OAuthCredentialStore,
        *,
        redirect_handler: Callable[[str], Awaitable[None]] | None = None,
        callback_handler: Callable[[], Awaitable["AuthorizationCodeResult"]] | None = None,
        client_metadata_url: str | None = None,
        validate_resource_url: Callable[[str, str | None], Awaitable[None]] | None = None,
        request_sender: OAuthRequestSender | None = None,
        on_event: OAuthEventHandler | None = None,
        refresh_lock: Callable[[], AbstractAsyncContextManager[bool]] | None = None,
        pending_ttl: float = _DEFAULT_PENDING_TTL_SECONDS,
        expiry_skew: float = _DEFAULT_EXPIRY_SKEW_SECONDS,
    ) -> None:
        """Create a provider for one MCP server and one credential store.

        Args:
            server_url: The MCP server URL.
            client_metadata: Metadata for dynamic client registration. Its
                first redirect URI receives the authorization callback.
            store: Where credentials are loaded from and saved to.
            redirect_handler: SDK handler that shows an authorization URL, for
                in-process interactive logins.
            callback_handler: SDK handler that waits for the authorization
                callback, for in-process interactive logins.
            client_metadata_url: URL-based client ID, passed to the SDK.
            validate_resource_url: Resource URL validator, passed to the SDK.
            request_sender: Sends discovery, registration, and token requests.
                Defaults to a fresh client per request that follows no
                redirects, times out after 30 seconds, and reads at most 1 MiB.
            on_event: Receives `OAuthEvent`s. An exception it raises propagates
                in place of the provider's own error, so a host can raise its own
                reauthorization error from `AuthorizationRequired`.
            refresh_lock: Returns a context manager that serializes refreshes
                across processes and yields whether the lock was acquired.
                Under the lock, credentials are reloaded so a peer's refresh is
                reused; without the lock, the refresh is skipped so a rotating
                refresh token is never replayed.
            pending_ttl: Seconds a `PendingAuthorization` stays valid.
            expiry_skew: Seconds before expiry that a token counts as expired.
        """
        adapter = _StoreAdapter(store)
        super().__init__(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=adapter,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            client_metadata_url=client_metadata_url,
            validate_resource_url=validate_resource_url,
        )
        adapter.provider = self
        self._store = store
        self._send = request_sender or _send_with_default_client
        self._on_event = on_event
        self._refresh_lock = refresh_lock
        self._pending_ttl = pending_ttl
        self._expiry_skew = expiry_skew
        self._expires_at: float | None = None
        self._pausing = False
        self._state_prefix = ""
        _omit_client_id_under_basic_auth(self)

    async def begin(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        state_prefix: str = "",
    ) -> AuthorizationStart:
        """Start a new login and return where to send the user.

        Probes the server without credentials, then runs the SDK's discovery
        and registration through `request_sender`, saving the registered
        client and metadata to the store. Existing tokens are ignored, not
        deleted.

        Args:
            headers: Extra headers for the unauthenticated probe of the MCP
                server, such as a static API key it also requires. They are
                not sent to the authorization server.
            state_prefix: Prepended to the random `state`, so a callback route
                can find the connection it belongs to. Not secret.

        Returns:
            The authorization URL and the pending state to persist.

        Raises:
            MCPOAuthError: If the server does not ask for OAuth, or discovery
                or registration fails. SDK errors, which may quote provider
                responses, are replaced; exceptions from `request_sender` and
                the store propagate unchanged.
            MCPOAuthUnavailableError: If a server cannot be reached.
        """
        probe = _initialize_request(self.context.server_url, headers)
        with _in_flow():
            await self._initialize()
            self.context.clear_tokens()
            self._pausing, self._state_prefix = True, state_prefix
            try:
                await self._drive_until_paused(probe)
            except _AuthorizationPausedError as paused:
                return paused.start
            except _UNREACHABLE:
                msg = "Could not reach the MCP server or its authorization server"
                raise MCPOAuthUnavailableError(msg) from None
            except (OAuthFlowError, ValidationError) as error:
                msg = f"MCP OAuth discovery or registration failed ({type(error).__name__})"
                raise MCPOAuthError(msg) from None
            finally:
                self._pausing, self._state_prefix = False, ""
        msg = "The MCP server did not request OAuth authorization"
        raise MCPOAuthError(msg)

    async def complete(
        self,
        pending: PendingAuthorization,
        *,
        code: str,
        state: str,
        iss: str | None = None,
    ) -> None:
        """Finish a login from its callback and save the resulting grant.

        Args:
            pending: The state `begin` returned.
            code: The `code` callback parameter.
            state: The `state` callback parameter.
            iss: The RFC 9207 `iss` callback parameter, if present.

        Raises:
            MCPOAuthError: If the callback does not match `pending`, `pending`
                expired, or the token endpoint rejected the code.
            MCPOAuthUnavailableError: If the token endpoint cannot be reached.
        """
        try:
            with _in_flow():
                await self._complete(pending, code=code, state=state, iss=iss)
        except Exception as error:
            await self._emit(AuthorizationFailed(self.context.server_url, _failure_reason(error)))
            raise
        await self._emit(AuthorizationCompleted(self.context.server_url))

    async def refresh(self) -> None:
        """Exchange the stored refresh token for a new grant and save it.

        Runs without `refresh_lock`; hosts that call this directly hold their
        own lock around it.

        Raises:
            MCPAuthorizationRequiredError: If there is no refresh token or the
                authorization server rejected it. Delivers `ReconnectRequired`.
            MCPOAuthUnavailableError: If the authorization server is
                unreachable or failed temporarily.
        """
        with _in_flow():
            if not self._initialized:
                await self._initialize()
            try:
                await self._refresh()
            except MCPAuthorizationRequiredError:
                await self._emit(ReconnectRequired(self.context.server_url))
                raise

    async def async_auth_flow(
        self, request: httpx2.Request
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        """Authenticate one MCP request, refreshing or authorizing as needed.

        Yields:
            The MCP request; authorization-server requests go to `request_sender`.

        Raises:
            MCPAuthorizationRequiredError: If the user must authorize and no
                in-process handlers can collect the callback.
        """
        async with self.context.lock:
            with _in_flow():
                if not self._initialized:
                    await self._initialize()
                self.context.protocol_version = request.headers.get(MCP_PROTOCOL_VERSION_HEADER)
                await self._refresh_before_request()
        flow = super().async_auth_flow(request)
        try:
            with _in_flow():
                outbound = await anext(flow)
            while True:
                if outbound is request:
                    response = yield outbound
                else:
                    response = await self._send(outbound)
                with _in_flow():
                    outbound = await flow.asend(response)
        except StopAsyncIteration:
            return
        except _AuthorizationPausedError as paused:
            start = paused.start
            await self._emit(
                AuthorizationRequired(
                    self.context.server_url, start.authorization_url, start.pending
                )
            )
            msg = "Authorization is required to use this MCP server"
            raise MCPAuthorizationRequiredError(msg) from None
        finally:
            await flow.aclose()

    async def _initialize(self) -> None:
        """Load the stored snapshot, including absolute expiry and metadata."""
        credentials = await self._store.load() or OAuthCredentials()
        context = self.context
        context.current_tokens = credentials.tokens
        context.client_info = credentials.client_info
        context.oauth_metadata = credentials.oauth_metadata or context.oauth_metadata
        context.protected_resource_metadata = (
            credentials.protected_resource_metadata or context.protected_resource_metadata
        )
        context.auth_server_url = credentials.auth_server_url or context.auth_server_url
        self._expires_at = credentials.expires_at
        context.token_expiry_time = self._expiry_time(credentials.expires_at)
        self._initialized = True

    def _expiry_time(self, expires_at: float | None) -> float | None:
        """Return the SDK's validity deadline for an absolute expiry.

        Floored at 1.0 because the SDK treats a falsy deadline as no expiry.
        """
        return None if expires_at is None else max(expires_at - self._expiry_skew, 1.0)

    async def _save(self, *, tokens: OAuthToken | None = None) -> None:
        """Save a full snapshot of the current credentials."""
        context = self.context
        if tokens is not None:
            self._expires_at = (
                None if tokens.expires_in is None else time.time() + tokens.expires_in
            )
            context.token_expiry_time = self._expiry_time(self._expires_at)
        current = tokens or context.current_tokens
        await self._store.save(
            OAuthCredentials(
                tokens=current,
                expires_at=self._expires_at if current is not None else None,
                client_info=context.client_info,
                oauth_metadata=context.oauth_metadata,
                protected_resource_metadata=context.protected_resource_metadata,
                auth_server_url=context.auth_server_url,
            )
        )

    async def _perform_authorization_code_grant(self) -> tuple[str, str]:
        """Use the SDK's interactive handlers, or pause with a resumable URL."""
        context = self.context
        if not self._pausing and context.redirect_handler and context.callback_handler:
            return await super()._perform_authorization_code_grant()
        raise _AuthorizationPausedError(self._authorization_start())

    def _authorization_start(self) -> AuthorizationStart:
        """Build the authorization request the way the SDK does, keeping its state."""
        context = self.context
        if not context.client_metadata.redirect_uris:
            msg = "OAuth client metadata has no redirect URI"
            raise MCPOAuthError(msg)
        if context.client_info is None:
            msg = "OAuth client registration did not complete"
            raise MCPOAuthError(msg)
        metadata = context.oauth_metadata
        if metadata is not None and metadata.authorization_endpoint:
            endpoint = str(metadata.authorization_endpoint)
        else:
            base = context.get_authorization_base_url(context.server_url)
            endpoint = urljoin(base, "/authorize")
        pkce = PKCEParameters.generate()
        pending = PendingAuthorization(
            state=f"{self._state_prefix}{secrets.token_urlsafe(32)}",
            code_verifier=pkce.code_verifier,
            redirect_uri=str(context.client_metadata.redirect_uris[0]),
            scope=context.client_metadata.scope,
            protocol_version=context.protocol_version,
            expires_at=time.time() + self._pending_ttl,
        )
        parameters = {
            "response_type": "code",
            "client_id": context.client_info.client_id,
            "redirect_uri": pending.redirect_uri,
            "state": pending.state,
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": "S256",
        }
        if context.should_include_resource_param(context.protocol_version):
            parameters["resource"] = context.get_resource_url()
        if pending.scope:
            parameters["scope"] = pending.scope
            if "offline_access" in pending.scope.split():
                parameters["prompt"] = "consent"
        separator = "&" if "?" in endpoint else "?"
        return AuthorizationStart(f"{endpoint}{separator}{urlencode(parameters)}", pending)

    async def _drive_until_paused(self, probe: httpx2.Request) -> None:
        """Run the SDK flow for `probe`, sending every request through the sender."""
        flow = super().async_auth_flow(probe)
        try:
            outbound = await anext(flow)
            while True:
                response = await self._send(outbound)
                if outbound is probe and response.status_code not in _CHALLENGES:
                    return
                outbound = await flow.asend(response)
        except StopAsyncIteration:
            return
        finally:
            await flow.aclose()

    async def _complete(
        self, pending: PendingAuthorization, *, code: str, state: str, iss: str | None
    ) -> None:
        if pending.expires_at <= time.time():
            msg = "The MCP authorization expired; start again"
            raise _CallbackError(msg, "expired")
        if not secrets.compare_digest(state.encode(), pending.state.encode()) or not code:
            msg = "The MCP authorization callback did not match this login"
            raise _CallbackError(msg, "invalid_callback")
        await self._initialize()
        context = self.context
        redirect_uris = context.client_metadata.redirect_uris or []
        if not redirect_uris or str(redirect_uris[0]) != pending.redirect_uri:
            msg = "The MCP authorization used a different redirect URI"
            raise _CallbackError(msg, "invalid_callback")
        if context.client_info is None:
            msg = "The registered OAuth client for this login is missing"
            raise _CallbackError(msg, "invalid_callback")
        try:
            validate_authorization_response_iss(iss, context.oauth_metadata)
        except OAuthFlowError:
            msg = "The MCP authorization callback came from an unexpected issuer"
            raise _CallbackError(msg, "invalid_callback") from None
        context.client_metadata.scope = pending.scope
        context.protocol_version = pending.protocol_version
        request = await self._exchange_token_authorization_code(code, pending.code_verifier)
        response = await self._send_or_unavailable(request)
        if response.status_code not in _TOKEN_SUCCESS:
            msg = f"The token endpoint rejected the authorization code: HTTP {response.status_code}"
            raise _CallbackError(msg + _error_code(response), "rejected")
        try:
            await self._handle_token_response(response)
        except (OAuthTokenError, ValidationError):
            msg = "The token endpoint returned an invalid token response"
            raise _CallbackError(msg, "rejected") from None

    async def _refresh(self) -> None:
        context = self.context
        tokens = context.current_tokens
        if tokens is None or not tokens.refresh_token or context.client_info is None:
            msg = "No refresh token is stored; connect this MCP server again"
            raise MCPAuthorizationRequiredError(msg)
        if context.oauth_metadata is None:
            await self._discover_metadata()
        response = await self._send_or_unavailable(await self._refresh_token())
        status = response.status_code
        if status == _TOO_MANY_REQUESTS or status >= _SERVER_ERROR:
            msg = f"The authorization server is temporarily unavailable: HTTP {status}"
            raise MCPOAuthUnavailableError(msg)
        if status != _OK:
            context.clear_tokens()
            msg = f"The authorization server rejected the refresh token: HTTP {status}"
            raise MCPAuthorizationRequiredError(msg + _error_code(response))
        if not await self._handle_refresh_response(response):
            msg = "The authorization server returned an invalid refresh response"
            raise MCPAuthorizationRequiredError(msg)

    async def _refresh_before_request(self) -> None:
        """Refresh an expired token, under the host's lock when one is given."""
        if not self._needs_refresh():
            return
        if self._refresh_lock is None:
            await self._refresh_or_fall_through()
            return
        async with self._refresh_lock() as acquired:
            await self._initialize()
            if not self._needs_refresh():
                return
            if acquired:
                await self._refresh_or_fall_through()
            else:
                self.context.clear_tokens()

    async def _refresh_or_fall_through(self) -> None:
        """Refresh, leaving a rejected grant to the SDK's reauthorization path."""
        try:
            await self._refresh()
        except MCPAuthorizationRequiredError:
            self.context.clear_tokens()

    def _needs_refresh(self) -> bool:
        return not self.context.is_token_valid() and self.context.can_refresh_token()

    async def _discover_metadata(self) -> None:
        """Discover authorization-server metadata that a refresh needs."""
        try:
            await self._discover_metadata_unchecked()
        except OAuthFlowError:
            msg = "Discovered OAuth metadata does not match this MCP server"
            raise MCPOAuthError(msg) from None

    async def _discover_metadata_unchecked(self) -> None:
        context = self.context
        for url in build_protected_resource_metadata_discovery_urls(None, context.server_url):
            response = await self._send_or_unavailable(create_oauth_metadata_request(url))
            resource = await handle_protected_resource_response(response)
            if resource is not None:
                await self._validate_resource_match(resource)
                context.protected_resource_metadata = resource
                context.auth_server_url = str(resource.authorization_servers[0])
                break
        for url in build_oauth_authorization_server_metadata_discovery_urls(
            context.auth_server_url, context.server_url
        ):
            response = await self._send_or_unavailable(create_oauth_metadata_request(url))
            found, metadata = await handle_auth_metadata_response(response)
            if not found:
                break
            if metadata is not None:
                if context.auth_server_url is not None:
                    validate_metadata_issuer(metadata, context.auth_server_url)
                context.oauth_metadata = metadata
                await self._save()
                return

    async def _send_or_unavailable(self, request: httpx2.Request) -> httpx2.Response:
        try:
            return await self._send(request)
        except _UNREACHABLE:
            msg = "Could not reach the MCP authorization server"
            raise MCPOAuthUnavailableError(msg) from None

    async def _emit(self, event: OAuthEvent) -> None:
        if self._on_event is not None:
            await self._on_event(event)


class _CallbackError(MCPOAuthError):
    def __init__(self, message: str, reason: AuthorizationFailureReason) -> None:
        super().__init__(message)
        self.reason: AuthorizationFailureReason = reason


def _failure_reason(error: Exception) -> AuthorizationFailureReason:
    if isinstance(error, _CallbackError):
        return error.reason
    if isinstance(error, MCPOAuthUnavailableError):
        return "unavailable"
    return "error"


def _initialize_request(server_url: str, headers: Mapping[str, str] | None) -> httpx2.Request:
    """Build the unauthenticated `initialize` call that elicits the 401 challenge."""
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "langchain", "version": "1"},
        },
    }
    return httpx2.Request(
        "POST",
        server_url,
        headers={
            **(headers or {}),
            "Accept": "application/json, text/event-stream",
            MCP_PROTOCOL_VERSION_HEADER: LATEST_PROTOCOL_VERSION,
        },
        json=payload,
    )


def _omit_client_id_under_basic_auth(provider: OAuthClientProvider) -> None:
    """Drop the body `client_id` from token requests authenticated with HTTP Basic.

    The SDK sends `client_id` in the body and in the Basic header. RFC 6749
    §2.3.1 carries the identity in the header, and some authorization servers
    reject the duplicate.
    """
    original = provider.context.prepare_token_auth

    def prepare_token_auth(
        data: dict[str, str], headers: dict[str, str] | None = None
    ) -> tuple[dict[str, str], dict[str, str]]:
        data, prepared = original(data, headers)
        if prepared.get("Authorization", "").lower().startswith("basic "):
            data = {key: value for key, value in data.items() if key != "client_id"}
        return data, prepared

    provider.context.prepare_token_auth = prepare_token_auth  # type: ignore[method-assign]

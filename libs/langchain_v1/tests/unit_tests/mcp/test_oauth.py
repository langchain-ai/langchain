"""Tests for `langchain.mcp.oauth`."""

# ruff: noqa: S105, S106  # every token and secret here is a fake fixture value

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from langchain.mcp.oauth import (
    AuthorizationCompleted,
    AuthorizationFailed,
    AuthorizationRequired,
    InMemoryOAuthCredentialStore,
    MCPAuthorizationRequiredError,
    MCPOAuthError,
    MCPOAuthProvider,
    MCPOAuthUnavailableError,
    OAuthCredentials,
    OAuthEvent,
    PendingAuthorization,
    ReconnectRequired,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

MCP_URL = "https://mcp.example.com/mcp"
ISSUER = "https://auth.example.com"
TOKEN_URL = f"{ISSUER}/oauth/token"
REDIRECT_URI = "https://app.example.com/oauth/callback"
SECRET_BODY = "provider-detail-with-secret"


@dataclass
class FakeServers:
    """An MCP resource server and its authorization server."""

    valid_tokens: set[str] = field(default_factory=set)
    challenges: dict[str, str] = field(default_factory=dict)
    issued: int = 0
    token_status: int = 200
    refresh_status: int = 200
    registration_status: int = 201
    auth_requests: list[httpx2.Request] = field(default_factory=list)
    resource_requests: list[httpx2.Request] = field(default_factory=list)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        if url == MCP_URL:
            self.resource_requests.append(request)
            bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
            if bearer in self.valid_tokens:
                return httpx2.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})
            challenge = (
                'Bearer resource_metadata="https://mcp.example.com/'
                '.well-known/oauth-protected-resource/mcp"'
            )
            return httpx2.Response(401, headers={"WWW-Authenticate": challenge})
        self.auth_requests.append(request)
        if url == "https://mcp.example.com/.well-known/oauth-protected-resource/mcp":
            return httpx2.Response(
                200, json={"resource": MCP_URL, "authorization_servers": [ISSUER]}
            )
        if url == f"{ISSUER}/.well-known/oauth-authorization-server":
            return httpx2.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{ISSUER}/oauth/authorize",
                    "token_endpoint": TOKEN_URL,
                    "registration_endpoint": f"{ISSUER}/oauth/register",
                    "response_types_supported": ["code"],
                    "code_challenge_methods_supported": ["S256"],
                },
            )
        if url == f"{ISSUER}/oauth/register":
            if self.registration_status != 201:
                return httpx2.Response(self.registration_status, text=SECRET_BODY)
            body = json.loads(request.content)
            return httpx2.Response(201, json={**body, "client_id": "client-1"})
        if url == TOKEN_URL:
            return self._token(request)
        return httpx2.Response(404)

    def _token(self, request: httpx2.Request) -> httpx2.Response:
        form = {key: values[0] for key, values in parse_qs(request.content.decode()).items()}
        if form["grant_type"] == "authorization_code":
            if self.token_status != 200:
                return httpx2.Response(
                    self.token_status,
                    json={"error": "invalid_grant", "error_description": SECRET_BODY},
                )
            digest = hashlib.sha256(form["code_verifier"].encode()).digest()
            challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            assert self.challenges.pop(form["code"]) == challenge
        elif self.refresh_status != 200:
            return httpx2.Response(self.refresh_status, json={"error": "invalid_grant"})
        self.issued += 1
        access = f"access-{self.issued}"
        self.valid_tokens.add(access)
        return httpx2.Response(
            200,
            json={
                "access_token": access,
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": f"refresh-{self.issued}",
            },
        )

    async def send(self, request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        return self.handle(request)

    def authorize(self, authorization_url: str) -> tuple[str, str]:
        """Approve an authorization request, returning its `code` and `state`."""
        query = parse_qs(urlsplit(authorization_url).query)
        code = f"code-{len(self.challenges)}"
        self.challenges[code] = query["code_challenge"][0]
        return code, query["state"][0]


def _metadata() -> OAuthClientMetadata:
    return OAuthClientMetadata(
        client_name="Test",
        redirect_uris=[AnyUrl(REDIRECT_URI)],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )


def _provider(
    servers: FakeServers,
    store: InMemoryOAuthCredentialStore,
    events: list[OAuthEvent] | None = None,
    **kwargs: Any,
) -> MCPOAuthProvider:
    async def on_event(event: OAuthEvent) -> None:
        if events is not None:
            events.append(event)

    return MCPOAuthProvider(
        MCP_URL, _metadata(), store, request_sender=servers.send, on_event=on_event, **kwargs
    )


def _client(servers: FakeServers, provider: MCPOAuthProvider) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(servers.handle),
        auth=provider,
        headers={"X-Resource-Key": "resource-secret"},
    )


async def _connect(servers: FakeServers, store: InMemoryOAuthCredentialStore) -> None:
    start = await _provider(servers, store).begin()
    code, state = servers.authorize(start.authorization_url)
    await _provider(servers, store).complete(start.pending, code=code, state=state, iss=ISSUER)


async def test_begin_returns_a_resumable_authorization() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()

    start = await _provider(servers, store).begin(
        headers={"X-Resource-Key": "probe"}, state_prefix="github."
    )

    query = parse_qs(urlsplit(start.authorization_url).query)
    assert start.authorization_url.startswith(f"{ISSUER}/oauth/authorize?")
    assert query["client_id"] == ["client-1"]
    assert query["redirect_uri"] == [REDIRECT_URI]
    assert query["code_challenge_method"] == ["S256"]
    assert query["resource"] == [MCP_URL]
    assert query["state"] == [start.pending.state]
    assert start.pending.state.startswith("github.")
    assert store.credentials is not None
    assert store.credentials.client_info is not None
    assert store.credentials.oauth_metadata is not None
    assert str(store.credentials.oauth_metadata.token_endpoint) == TOKEN_URL
    (probe,) = servers.resource_requests
    assert probe.headers["x-resource-key"] == "probe"
    assert "authorization" not in probe.headers
    assert all("x-resource-key" not in request.headers for request in servers.auth_requests)


async def test_complete_resumes_in_a_new_provider() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    events: list[OAuthEvent] = []
    start = await _provider(servers, store).begin()
    pending = PendingAuthorization.model_validate_json(start.pending.model_dump_json())
    code, state = servers.authorize(start.authorization_url)
    before = time.time()

    await _provider(servers, store, events).complete(pending, code=code, state=state, iss=ISSUER)

    credentials = store.credentials
    assert credentials is not None
    assert credentials.tokens is not None
    assert credentials.tokens.access_token == "access-1"
    assert credentials.expires_at is not None
    assert before + 3600 <= credentials.expires_at <= time.time() + 3600
    assert events == [AuthorizationCompleted(MCP_URL)]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"state": "forged"}, "invalid_callback"),
        ({"iss": "https://evil.example.com"}, "invalid_callback"),
        ({"expired": True}, "expired"),
    ],
)
async def test_complete_rejects_a_mismatched_callback(change: dict[str, Any], reason: str) -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    events: list[OAuthEvent] = []
    start = await _provider(servers, store).begin()
    code, state = servers.authorize(start.authorization_url)
    pending = start.pending
    if change.pop("expired", False):
        pending = pending.model_copy(update={"expires_at": time.time() - 1})
    arguments = {"code": code, "state": state, "iss": ISSUER, **change}

    with pytest.raises(MCPOAuthError):
        await _provider(servers, store, events).complete(pending, **arguments)

    assert events == [AuthorizationFailed(MCP_URL, reason)]  # type: ignore[arg-type]
    assert store.credentials is not None
    assert store.credentials.tokens is None


async def test_token_errors_name_the_code_but_not_the_body() -> None:
    servers, store = FakeServers(token_status=400), InMemoryOAuthCredentialStore()
    start = await _provider(servers, store).begin()
    code, state = servers.authorize(start.authorization_url)

    with pytest.raises(MCPOAuthError) as caught:
        await _provider(servers, store).complete(start.pending, code=code, state=state)

    assert "HTTP 400 (invalid_grant)" in str(caught.value)
    assert SECRET_BODY not in str(caught.value)
    assert caught.value.__cause__ is None


async def test_expired_token_refreshes_at_the_stored_endpoint_after_restart() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    await _connect(servers, store)
    assert store.credentials is not None
    store.credentials = store.credentials.model_copy(update={"expires_at": time.time() - 60})
    servers.auth_requests.clear()

    async with _client(servers, _provider(servers, store)) as client:
        response = await client.post(MCP_URL, json={})

    assert response.status_code == 200
    assert [str(request.url) for request in servers.auth_requests] == [TOKEN_URL]
    assert servers.resource_requests[-1].headers["authorization"] == "Bearer access-2"
    assert "x-resource-key" not in servers.auth_requests[0].headers
    assert store.credentials.tokens is not None
    assert store.credentials.tokens.refresh_token == "refresh-2"


async def test_refresh_discovers_the_token_endpoint_when_none_is_stored() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    await _connect(servers, store)
    assert store.credentials is not None
    store.credentials = store.credentials.model_copy(
        update={"oauth_metadata": None, "protected_resource_metadata": None}
    )

    await _provider(servers, store).refresh()

    assert str(servers.auth_requests[-1].url) == TOKEN_URL
    assert store.credentials.oauth_metadata is not None


@pytest.mark.parametrize(
    ("status", "error", "event"),
    [
        (400, MCPAuthorizationRequiredError, [ReconnectRequired(MCP_URL)]),
        (503, MCPOAuthUnavailableError, []),
    ],
)
async def test_refresh_failures_distinguish_revocation_from_outages(
    status: int, error: type[Exception], event: list[OAuthEvent]
) -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    await _connect(servers, store)
    servers.refresh_status = status
    events: list[OAuthEvent] = []

    with pytest.raises(error):
        await _provider(servers, store, events).refresh()

    assert events == event


async def test_request_needing_authorization_hands_the_url_to_the_host() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    events: list[OAuthEvent] = []

    async with _client(servers, _provider(servers, store, events)) as client:
        with pytest.raises(MCPAuthorizationRequiredError) as caught:
            await client.post(MCP_URL, json={})

    (event,) = events
    assert isinstance(event, AuthorizationRequired)
    assert event.authorization_url.startswith(f"{ISSUER}/oauth/authorize?")
    assert event.authorization_url not in str(caught.value)
    assert event.authorization_url not in repr(event)
    code, state = servers.authorize(event.authorization_url)
    await _provider(servers, store).complete(event.pending, code=code, state=state)
    async with _client(servers, _provider(servers, store)) as client:
        assert (await client.post(MCP_URL, json={})).status_code == 200


async def test_basic_auth_omits_the_duplicate_body_client_id() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    await _connect(servers, store)
    assert store.credentials is not None
    assert store.credentials.client_info is not None
    client_info = store.credentials.client_info.model_copy(
        update={"token_endpoint_auth_method": "client_secret_basic", "client_secret": "s3cret"}
    )
    store.credentials = store.credentials.model_copy(update={"client_info": client_info})

    await _provider(servers, store).refresh()

    request = servers.auth_requests[-1]
    assert request.headers["authorization"].startswith("Basic ")
    assert "client_id" not in parse_qs(request.content.decode())


async def test_sdk_logs_omit_provider_response_bodies(caplog: pytest.LogCaptureFixture) -> None:
    servers, store = FakeServers(registration_status=400), InMemoryOAuthCredentialStore()
    caplog.set_level(logging.DEBUG)

    async with _client(servers, _provider(servers, store)) as client:
        with pytest.raises(Exception, match="Registration failed"):
            await client.post(MCP_URL, json={})

    assert "OAuth flow error" in caplog.text
    assert SECRET_BODY not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


async def test_refresh_lock_reuses_a_peer_refresh() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    await _connect(servers, store)
    assert store.credentials is not None
    stale = store.credentials.model_copy(update={"expires_at": time.time() - 60})
    fresh = store.credentials
    store.credentials = stale
    provider = _provider(servers, store)

    @contextlib.asynccontextmanager
    async def peer_refreshed() -> AsyncIterator[bool]:
        store.credentials = fresh
        yield True

    provider._refresh_lock = peer_refreshed
    servers.auth_requests.clear()
    async with _client(servers, provider) as client:
        assert (await client.post(MCP_URL, json={})).status_code == 200

    assert servers.auth_requests == []


async def test_refresh_lock_timeout_never_replays_the_refresh_token() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    await _connect(servers, store)
    assert store.credentials is not None
    store.credentials = store.credentials.model_copy(update={"expires_at": time.time() - 60})

    @contextlib.asynccontextmanager
    async def timed_out() -> AsyncIterator[bool]:
        yield False

    servers.auth_requests.clear()
    async with _client(servers, _provider(servers, store, refresh_lock=timed_out)) as client:
        with pytest.raises(MCPAuthorizationRequiredError):
            await client.post(MCP_URL, json={})

    assert TOKEN_URL not in [str(request.url) for request in servers.auth_requests]


async def test_begin_rejects_a_server_that_does_not_ask_for_oauth() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()

    async def open_server(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={}, request=request)

    provider = MCPOAuthProvider(MCP_URL, _metadata(), store, request_sender=open_server)

    with pytest.raises(MCPOAuthError, match="did not request OAuth"):
        await provider.begin()
    assert servers.auth_requests == []


def test_reprs_hide_credentials() -> None:
    credentials = OAuthCredentials(
        tokens=OAuthToken(access_token="access-secret", refresh_token="refresh-secret"),
        client_info=OAuthClientInformationFull(
            client_id="client", client_secret="client-secret", redirect_uris=[AnyUrl(REDIRECT_URI)]
        ),
    )
    pending = PendingAuthorization(
        state="state-secret",
        code_verifier="verifier-secret",
        redirect_uri=REDIRECT_URI,
        expires_at=0,
    )

    rendered = repr(credentials) + repr(pending)

    assert "secret" not in rendered.replace("client_secret", "")


async def test_inline_refresh_sends_resource_for_the_request_protocol_version() -> None:
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    await _connect(servers, store)
    assert store.credentials is not None
    store.credentials = store.credentials.model_copy(
        update={"expires_at": time.time() - 60, "protected_resource_metadata": None}
    )
    servers.auth_requests.clear()

    async with _client(servers, _provider(servers, store)) as client:
        await client.post(MCP_URL, json={}, headers={"MCP-Protocol-Version": "2025-11-25"})

    (refresh,) = servers.auth_requests
    assert parse_qs(refresh.content.decode())["resource"] == [MCP_URL]


async def test_event_handler_can_raise_the_host_error() -> None:
    class HostReauthError(Exception):
        pass

    servers, store = FakeServers(), InMemoryOAuthCredentialStore()

    async def on_event(event: OAuthEvent) -> None:
        if isinstance(event, AuthorizationRequired):
            raise HostReauthError

    provider = MCPOAuthProvider(
        MCP_URL, _metadata(), store, request_sender=servers.send, on_event=on_event
    )

    async with _client(servers, provider) as client:
        with pytest.raises(HostReauthError):
            await client.post(MCP_URL, json={})


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (httpx2.ConnectError("refused"), MCPOAuthUnavailableError),
        (httpx2.ReadTimeout("timed out"), httpx2.ReadTimeout),
    ],
)
async def test_refresh_reports_only_unsent_requests_as_unavailable(
    failure: Exception, expected: type[Exception]
) -> None:
    """A read timeout may follow a rotation the server already committed."""
    servers, store = FakeServers(), InMemoryOAuthCredentialStore()
    await _connect(servers, store)

    async def failing(_request: httpx2.Request) -> httpx2.Response:
        raise failure

    provider = MCPOAuthProvider(MCP_URL, _metadata(), store, request_sender=failing)

    with pytest.raises(expected):
        await provider.refresh()


async def test_begin_propagates_host_sender_errors() -> None:
    class BlockedError(Exception):
        pass

    servers, store = FakeServers(), InMemoryOAuthCredentialStore()

    async def guarded(request: httpx2.Request) -> httpx2.Response:
        if str(request.url).startswith(ISSUER):
            msg = "blocked by host policy"
            raise BlockedError(msg)
        return await servers.send(request)

    provider = MCPOAuthProvider(MCP_URL, _metadata(), store, request_sender=guarded)

    with pytest.raises(BlockedError, match="host policy"):
        await provider.begin()

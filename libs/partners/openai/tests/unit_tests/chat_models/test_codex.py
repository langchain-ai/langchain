"""Unit tests for `_ChatOpenAICodex`."""
# ruff: noqa: S106, S107

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage

import langchain_openai.chat_models.codex as codex_module
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.chat_models.codex import (
    ACCOUNT_ID_HEADER,
    CHATGPT_CODEX_BASE_URL,
    DEFAULT_INSTRUCTIONS,
    EXPERIMENTAL_UNOFFICIAL_WARNING,
    ORIGINATOR_ENV_VAR,
    ORIGINATOR_HEADER,
    ORIGINATOR_VALUE,
    SESSION_ID_HEADER,
    _ChatOpenAICodex,
    _SyncTokenCallable,
)
from langchain_openai.chatgpt_oauth import _ChatGPTToken


class FakeTokenProvider:
    """Minimal `_ChatGPTOAuthTokenProvider` for tests."""

    def __init__(
        self,
        access_token: str = "at-1",
        account_id: str | None = "acct-1",
    ) -> None:
        self.access_token = access_token
        self.account_id = account_id
        self.calls = 0
        self.async_calls = 0

    def get_token(self) -> _ChatGPTToken:
        self.calls += 1
        return _ChatGPTToken(
            access_token=self.access_token,
            refresh_token="rt",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            account_id=self.account_id,
        )

    async def aget_token(self) -> _ChatGPTToken:
        self.async_calls += 1
        return self.get_token()

    def get_access_token(self) -> str:
        return self.get_token().access_token

    async def aget_access_token(self) -> str:
        token = await self.aget_token()
        return token.access_token


class AsyncOnlyTokenProvider(FakeTokenProvider):
    """Provider whose async path never delegates to sync `get_token`.

    Lets a test prove an async code path stayed off the event loop: any stray
    sync `get_token()` would bump `calls`, which the async path must leave
    untouched.
    """

    async def aget_token(self) -> _ChatGPTToken:
        self.async_calls += 1
        return _ChatGPTToken(
            access_token=self.access_token,
            refresh_token="rt",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            account_id=self.account_id,
        )


def _build_model(**overrides: Any) -> _ChatOpenAICodex:
    provider = overrides.pop("token_provider", None) or FakeTokenProvider()
    return _ChatOpenAICodex(
        model=overrides.pop("model", "gpt-5.2-codex"),
        token_provider=provider,
        **overrides,
    )


def test_experimental_unofficial_warning_is_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codex_module, "_experimental_warning_emitted", False)
    with pytest.warns(UserWarning, match="experimental and unofficial") as warning:
        _build_model()
    assert warning[0].filename == __file__
    assert "applicable OpenAI terms" in EXPERIMENTAL_UNOFFICIAL_WARNING


def test_defaults_route_to_chatgpt_codex_backend() -> None:
    model = _build_model()
    assert model.openai_api_base == CHATGPT_CODEX_BASE_URL
    assert model.use_responses_api is True
    assert model.store is False
    assert model.streaming is True
    # `output_version` is a client-side projection and isn't forced — the
    # base default applies unless the caller picks one explicitly.


def test_uses_callable_api_key_from_token_provider() -> None:
    """The SDK-facing `api_key` must resolve to the provider's current token."""
    provider = FakeTokenProvider(access_token="abc")
    model = _build_model(token_provider=provider)
    secret = model.openai_api_key
    assert secret is not None
    if callable(secret):
        assert secret() == "abc"
    else:
        # `ChatOpenAI.validate_environment` wraps callables in `SecretStr`.
        assert secret.get_secret_value() == "abc"
    assert provider.calls >= 1


@pytest.mark.parametrize("field", ["base_url", "openai_api_base"])
def test_base_url_override_is_rejected(field: str) -> None:
    """Reject caller-supplied `base_url` / `openai_api_base`.

    The OAuth bearer token is wired in as `api_key`, so accepting an arbitrary
    base URL would let an attacker (or a misconfigured serialized config)
    exfiltrate the token to a host of their choice.
    """
    with pytest.raises(ValueError, match=r"requires `(?:base_url|openai_api_base)="):
        _build_model(**{field: "https://attacker.example.com/codex"})


def test_explicit_base_url_matching_codex_endpoint_is_accepted() -> None:
    """Passing the canonical Codex endpoint explicitly is still allowed."""
    model = _build_model(base_url=CHATGPT_CODEX_BASE_URL)
    assert model.openai_api_base == CHATGPT_CODEX_BASE_URL


@pytest.mark.parametrize("field", ["api_key", "openai_api_key"])
def test_explicit_api_key_is_rejected(field: str) -> None:
    """Reject a caller-supplied `api_key` / `openai_api_key`.

    Auth is owned by `token_provider`; a caller-supplied key would silently
    win over the OAuth bearer, so it must fail loudly rather than leave the
    model in a conflicting state.
    """
    with pytest.raises(ValueError, match=r"manages authentication via"):
        _build_model(**{field: "sk-should-not-be-allowed"})


def test_request_payload_injects_account_id_and_originator_headers() -> None:
    provider = FakeTokenProvider(account_id="acct-42")
    model = _build_model(token_provider=provider)
    payload = model._get_request_payload([HumanMessage("hi")])
    headers = payload["extra_headers"]
    assert headers[ACCOUNT_ID_HEADER] == "acct-42"
    assert headers[ORIGINATOR_HEADER] == ORIGINATOR_VALUE


def test_request_payload_omits_account_id_when_unknown() -> None:
    provider = FakeTokenProvider(account_id=None)
    model = _build_model(token_provider=provider)
    payload = model._get_request_payload([HumanMessage("hi")])
    headers = payload["extra_headers"]
    assert ACCOUNT_ID_HEADER not in headers
    assert headers[ORIGINATOR_HEADER] == ORIGINATOR_VALUE


def test_request_payload_can_disable_originator_header() -> None:
    """`originator=None` omits the header entirely."""
    provider = FakeTokenProvider(account_id=None)
    model = _build_model(token_provider=provider, originator=None, session_id=None)
    payload = model._get_request_payload([HumanMessage("hi")])
    assert "extra_headers" not in payload


def test_session_id_header_is_stable_per_instance() -> None:
    """Every turn on one instance shares a `session-id`, so it can be cached."""
    model = _build_model()
    first = model._get_request_payload([HumanMessage("hi")])
    second = model._get_request_payload([HumanMessage("hi again")])
    session_id = first["extra_headers"][SESSION_ID_HEADER]
    assert session_id
    assert second["extra_headers"][SESSION_ID_HEADER] == session_id
    assert "session_id" not in first


def test_session_id_defaults_differ_across_instances() -> None:
    first = _build_model()._get_request_payload([HumanMessage("hi")])
    second = _build_model()._get_request_payload([HumanMessage("hi")])
    assert (
        first["extra_headers"][SESSION_ID_HEADER]
        != second["extra_headers"][SESSION_ID_HEADER]
    )


def test_constructor_session_id_is_sent() -> None:
    model = _build_model(session_id="thread-1")
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["extra_headers"][SESSION_ID_HEADER] == "thread-1"


def test_per_call_session_id_overrides_field_and_stays_out_of_body() -> None:
    model = _build_model(session_id="thread-1")
    payload = model._get_request_payload([HumanMessage("hi")], session_id="thread-2")
    assert payload["extra_headers"][SESSION_ID_HEADER] == "thread-2"
    assert "session_id" not in payload


def test_session_id_none_omits_header() -> None:
    model = _build_model(session_id=None)
    payload = model._get_request_payload([HumanMessage("hi")])
    assert SESSION_ID_HEADER not in payload["extra_headers"]


def test_caller_extra_headers_override_session_id() -> None:
    model = _build_model(session_id="thread-1")
    payload = model._get_request_payload(
        [HumanMessage("hi")], extra_headers={SESSION_ID_HEADER: "call-session"}
    )
    assert payload["extra_headers"][SESSION_ID_HEADER] == "call-session"


def test_session_id_is_added_to_prebuilt_async_headers() -> None:
    """The async paths hand in prebuilt headers; `session-id` must still ride."""
    provider = AsyncOnlyTokenProvider(account_id=None)
    model = _build_model(token_provider=provider, originator=None)
    payload = model._get_request_payload(
        [HumanMessage("hi")], _codex_headers={}, session_id="thread-async"
    )
    assert payload["extra_headers"] == {SESSION_ID_HEADER: "thread-async"}
    assert provider.calls == 0


def test_constructor_originator_overrides_default() -> None:
    """An explicit `originator=` constructor value replaces the package default."""
    model = _build_model(originator="my-app/1.2")
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["extra_headers"][ORIGINATOR_HEADER] == "my-app/1.2"


def test_env_var_overrides_default_originator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`LANGCHAIN_CODEX_ORIGINATOR` sets the default when no constructor value."""
    monkeypatch.setenv(ORIGINATOR_ENV_VAR, "env-app/2.0")
    model = _build_model()
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["extra_headers"][ORIGINATOR_HEADER] == "env-app/2.0"


def test_constructor_originator_wins_over_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit constructor value beats the env var (resolution order)."""
    monkeypatch.setenv(ORIGINATOR_ENV_VAR, "env-app/2.0")
    model = _build_model(originator="ctor-app/9.9")
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["extra_headers"][ORIGINATOR_HEADER] == "ctor-app/9.9"


def test_empty_env_var_falls_back_to_package_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty `LANGCHAIN_CODEX_ORIGINATOR` is treated as unset."""
    monkeypatch.setenv(ORIGINATOR_ENV_VAR, "")
    model = _build_model()
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["extra_headers"][ORIGINATOR_HEADER] == ORIGINATOR_VALUE


def test_caller_extra_headers_override_originator_field() -> None:
    """A per-call `extra_headers` originator beats the model's field value."""
    model = _build_model(originator="ctor-app")
    payload = model._get_request_payload(
        [HumanMessage("hi")], extra_headers={ORIGINATOR_HEADER: "call-app"}
    )
    assert payload["extra_headers"][ORIGINATOR_HEADER] == "call-app"


def test_request_payload_pulls_fresh_account_id_each_call() -> None:
    provider = FakeTokenProvider()
    model = _build_model(token_provider=provider)
    before = provider.calls
    model._get_request_payload([HumanMessage("hi")])
    model._get_request_payload([HumanMessage("hi again")])
    assert provider.calls >= before + 2


def test_invalid_token_provider_rejected() -> None:
    with pytest.raises(TypeError):
        _ChatOpenAICodex(model="gpt-5.2-codex", token_provider="not-a-provider")


def test_conflicting_use_responses_api_raises() -> None:
    with pytest.raises(ValueError, match="use_responses_api"):
        _ChatOpenAICodex(
            model="gpt-5.2-codex",
            token_provider=FakeTokenProvider(),
            use_responses_api=False,
        )


@pytest.mark.parametrize("output_version", ["v0", "responses/v1", "v1"])
def test_explicit_output_version_is_respected(output_version: str) -> None:
    """`output_version` is a client-side projection — any value is allowed.

    Also pins the wire-level invariant: regardless of the constructor
    choice, `output_version` never appears in the outbound payload (it's
    consumed entirely by the response projection layer).
    """
    model = _ChatOpenAICodex(
        model="gpt-5.2-codex",
        token_provider=FakeTokenProvider(),
        output_version=output_version,
    )
    assert model.output_version == output_version
    payload = model._get_request_payload([HumanMessage("hi")])
    assert "output_version" not in payload


def test_conflicting_store_raises() -> None:
    with pytest.raises(ValueError, match="store"):
        _ChatOpenAICodex(
            model="gpt-5.2-codex",
            token_provider=FakeTokenProvider(),
            store=True,
        )


def test_conflicting_streaming_raises() -> None:
    with pytest.raises(ValueError, match="streaming"):
        _ChatOpenAICodex(
            model="gpt-5.2-codex",
            token_provider=FakeTokenProvider(),
            streaming=False,
        )


def test_request_payload_sends_store_false_and_stream_true() -> None:
    """The Codex backend 400s with `store=True` or non-streaming requests."""
    model = _build_model()
    payload = model._get_request_payload([HumanMessage("hi")])
    assert payload["store"] is False
    assert payload["stream"] is True


def test_request_payload_with_store_false_drops_reasoning_item_references() -> None:
    model = _build_model(instructions="")
    payload = model._get_request_payload(
        [
            HumanMessage("Explain recursive file search."),
            AIMessage(
                content=[
                    {"type": "reasoning", "id": "rs_123", "summary": []},
                    {
                        "type": "reasoning",
                        "id": "rs_456",
                        "summary": [],
                        "encrypted_content": "encrypted-context",
                    },
                    {"type": "text", "text": "Use pathlib.rglob.", "id": "msg_123"},
                ],
                response_metadata={"id": "resp_123"},
            ),
            HumanMessage("Make it shorter."),
        ]
    )

    assert payload["store"] is False
    assert [item.get("type") for item in payload["input"]] == [
        "message",
        "reasoning",
        "message",
        "message",
    ]
    assert payload["input"][1] == {
        "type": "reasoning",
        "id": "rs_456",
        "summary": [],
        "encrypted_content": "encrypted-context",
    }
    assert payload["input"][2]["content"] == [
        {"type": "output_text", "text": "Use pathlib.rglob.", "annotations": []}
    ]
    assert "id" not in payload["input"][2]


def test_request_payload_with_store_false_drops_image_call_references() -> None:
    model = _build_model(instructions="")
    payload = model._get_request_payload(
        [
            HumanMessage("Draw a word."),
            AIMessage(
                content=[
                    {"type": "image_generation_call", "id": "ig_reference_only"},
                    {
                        "type": "image_generation_call",
                        "id": "ig_with_result",
                        "result": "base64-image",
                        "status": "completed",
                        "index": 0,
                    },
                    {"type": "text", "text": "Done.", "id": "msg_123"},
                ],
                response_metadata={"id": "resp_123"},
            ),
            HumanMessage("Change the color."),
        ]
    )

    assert payload["store"] is False
    assert [item.get("type") for item in payload["input"]] == [
        "message",
        "image_generation_call",
        "message",
        "message",
    ]
    assert payload["input"][1] == {
        "type": "image_generation_call",
        "id": "ig_with_result",
        "result": "base64-image",
        "status": "completed",
    }
    assert payload["input"][2]["content"] == [
        {"type": "output_text", "text": "Done.", "annotations": []}
    ]
    assert "id" not in payload["input"][2]


def _developer_message(content: str) -> dict[str, Any]:
    return {"type": "message", "role": "developer", "content": content}


def test_request_payload_sends_default_prompt_as_developer_message() -> None:
    """The default prompt leads `input` so GPT-6.2+ can prompt-cache it."""
    model = _build_model()
    payload = model._get_request_payload([HumanMessage("hi")])
    assert "instructions" not in payload
    assert payload["input"][0] == _developer_message(DEFAULT_INSTRUCTIONS)
    assert [entry.get("role") for entry in payload["input"]] == [
        "developer",
        "user",
    ]


def test_request_payload_sends_constructor_instructions_as_developer_message() -> None:
    model = _build_model(instructions="custom system prompt")
    payload = model._get_request_payload([HumanMessage("hi")])
    assert "instructions" not in payload
    assert payload["input"][0] == _developer_message("custom system prompt")


def test_request_payload_with_empty_constructor_instructions_sends_no_prompt() -> None:
    model = _build_model(instructions="")
    payload = model._get_request_payload([HumanMessage("hi")])
    assert "instructions" not in payload
    assert [entry.get("role") for entry in payload["input"]] == ["user"]


def test_request_payload_with_string_input_leads_with_developer_message() -> None:
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload("hi")
    assert payload["input"][0] == _developer_message("model-level")
    assert payload["input"][1]["role"] == "user"


def test_per_call_instructions_are_sent_alongside_developer_prompt() -> None:
    """A per-call `instructions` kwarg is turn-only guidance, not a replacement."""
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload(
        [HumanMessage("hi")], instructions="call-level"
    )
    assert payload["instructions"] == "call-level"
    assert payload["input"][0] == _developer_message("model-level")


def test_request_payload_preserves_explicit_empty_instructions() -> None:
    """An explicit empty `instructions=""` is passed through verbatim."""
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload([HumanMessage("hi")], instructions="")
    assert payload["instructions"] == ""


def test_system_message_is_sent_as_leading_developer_message() -> None:
    """`SystemMessage` content replaces the constructor prompt.

    Codex rejects `system`-role input entries (400 "System messages are not
    allowed"), and GPT-6.2+ does not cache top-level `instructions`, so the
    content is sent as the first `developer` message in `input`.
    """
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload(
        [SystemMessage("from-system-message"), HumanMessage("hi")]
    )
    assert "instructions" not in payload
    input_messages = payload["input"]
    assert input_messages[0] == _developer_message("from-system-message")
    assert all(entry.get("role") != "system" for entry in input_messages)
    assert [entry.get("role") for entry in input_messages] == ["developer", "user"]
    assert "hi" in str(input_messages[1].get("content"))


def test_developer_prompt_is_identical_across_turns() -> None:
    """The cacheable prefix must stay byte-stable as the conversation grows."""
    model = _build_model()
    first = model._get_request_payload([SystemMessage("prompt"), HumanMessage("a")])
    second = model._get_request_payload(
        [
            SystemMessage("prompt"),
            HumanMessage("a"),
            AIMessage("b"),
            HumanMessage("c"),
        ]
    )
    assert first["input"][:2] == second["input"][:2]


@pytest.mark.parametrize("role", ["system", "developer"])
def test_chat_message_instruction_roles_become_developer_message(role: str) -> None:
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload(
        [
            ChatMessage(content=f"from-{role}", role=role),
            HumanMessage("hi"),
        ]
    )
    assert "instructions" not in payload
    assert payload["input"][0] == _developer_message(f"from-{role}")
    assert [entry.get("role") for entry in payload["input"]] == ["developer", "user"]


def test_back_to_back_system_messages_join_in_input_order() -> None:
    """Adjacent `SystemMessage` entries are concatenated with `"\\n\\n"`."""
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload(
        [
            SystemMessage("first"),
            SystemMessage("second"),
            HumanMessage("hi"),
        ]
    )
    assert payload["input"][0] == _developer_message("first\n\nsecond")
    assert len(payload["input"]) == 2


def test_interleaved_system_messages_are_moved_to_the_front() -> None:
    """`SystemMessage`s anywhere in the input join the leading developer message.

    All `SystemMessage`s are concatenated in input order and stripped; the
    remaining (non-system) messages keep their relative order.
    """
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload(
        [
            HumanMessage("hi"),
            SystemMessage("midway-system"),
            HumanMessage("bye"),
            SystemMessage("tail-system"),
        ]
    )
    assert payload["input"][0] == _developer_message("midway-system\n\ntail-system")
    contents = [entry.get("content") for entry in payload["input"][1:]]
    assert all(entry.get("role") != "system" for entry in payload["input"])
    assert contents == ["hi", "bye"]


def test_explicit_instructions_kwarg_keeps_system_message() -> None:
    """A per-call `instructions=` kwarg no longer discards `SystemMessage`s."""
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload(
        [SystemMessage("from-system-message"), HumanMessage("hi")],
        instructions="from-kwarg",
    )
    assert payload["instructions"] == "from-kwarg"
    assert payload["input"][0] == _developer_message("from-system-message")


def test_system_message_with_text_blocks_is_flattened() -> None:
    """List-of-text-blocks `SystemMessage` content flattens cleanly."""
    model = _build_model(instructions="model-level")
    payload = model._get_request_payload(
        [
            SystemMessage(
                [
                    {"type": "text", "text": "part one. "},
                    {"type": "text", "text": "part two."},
                ]
            ),
            HumanMessage("hi"),
        ]
    )
    assert payload["input"][0] == _developer_message("part one. part two.")


def test_system_message_with_non_text_block_raises() -> None:
    """Non-text content blocks can't be flattened into the developer prompt."""
    model = _build_model(instructions="model-level")
    with pytest.raises(ValueError, match="non-text content block"):
        model._get_request_payload(
            [
                SystemMessage(
                    [
                        {"type": "text", "text": "ok"},
                        {"type": "image_url", "image_url": "http://example/x.png"},
                    ]
                ),
                HumanMessage("hi"),
            ]
        )


def test_system_message_with_non_string_text_value_raises() -> None:
    """A text block whose `text` isn't a string is a programming error."""
    from langchain_openai.chat_models.codex import _flatten_system_message_content

    # Constructed directly (bypassing the `SystemMessage` content type so
    # we can hit the helper's defensive type check).
    bad = SystemMessage.model_construct(content=[{"type": "text", "text": 42}])
    with pytest.raises(ValueError, match=r"text.*not a string"):
        _flatten_system_message_content([bad])


def test_system_message_with_unsupported_content_type_raises() -> None:
    """Content that's neither str nor list (e.g., int) is rejected upfront."""
    from langchain_openai.chat_models.codex import _flatten_system_message_content

    bad = SystemMessage.model_construct(content=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported content type"):
        _flatten_system_message_content([bad])


def test_caller_headers_win_over_codex_defaults() -> None:
    """Caller-supplied `extra_headers` must override the Codex injections."""
    provider = FakeTokenProvider(account_id="acct-1")
    model = _build_model(token_provider=provider)
    payload = model._get_request_payload(
        [HumanMessage("hi")],
        extra_headers={ORIGINATOR_HEADER: "custom-app"},
    )
    headers = payload["extra_headers"]
    assert headers[ORIGINATOR_HEADER] == "custom-app"
    # The auto-injected account ID still rides along when not overridden.
    assert headers[ACCOUNT_ID_HEADER] == "acct-1"


async def test_agenerate_builds_codex_headers_without_sync_token_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_agenerate` must not call sync `get_token` on the event loop."""
    provider = AsyncOnlyTokenProvider(account_id="acct-async")
    model = _build_model(token_provider=provider)
    kwargs_seen: dict[str, Any] = {}

    async def _fake_super_agenerate(*_a: Any, **kwargs: Any) -> Any:
        kwargs_seen.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(ChatOpenAI, "_agenerate", _fake_super_agenerate)
    before_sync = provider.calls
    before_async = provider.async_calls
    result = await model._agenerate([HumanMessage("hi")])

    assert result == "sentinel"
    assert provider.async_calls == before_async + 1
    assert provider.calls == before_sync
    assert kwargs_seen["_codex_headers"] == {
        ACCOUNT_ID_HEADER: "acct-async",
        ORIGINATOR_HEADER: ORIGINATOR_VALUE,
    }


async def test_astream_builds_codex_headers_without_sync_token_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_astream` must not call sync `get_token` on the event loop."""
    provider = AsyncOnlyTokenProvider(account_id="acct-async")
    model = _build_model(token_provider=provider)
    kwargs_seen: dict[str, Any] = {}

    async def _fake_super_astream(*_a: Any, **kwargs: Any) -> Any:
        kwargs_seen.update(kwargs)
        yield "chunk"

    monkeypatch.setattr(ChatOpenAI, "_astream", _fake_super_astream)
    before_sync = provider.calls
    before_async = provider.async_calls
    received = [chunk async for chunk in model._astream([HumanMessage("hi")])]

    assert received == ["chunk"]
    assert provider.async_calls == before_async + 1
    assert provider.calls == before_sync
    assert kwargs_seen["_codex_headers"] == {
        ACCOUNT_ID_HEADER: "acct-async",
        ORIGINATOR_HEADER: ORIGINATOR_VALUE,
    }


async def test_astream_empty_headers_skip_sync_token_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accountless token yields `{}` headers — still no sync read on the loop.

    Guards the deliberate `is not None` check in `_get_request_payload`: an
    explicitly-built empty dict must be honored as-is, not treated as falsy and
    routed back through the blocking sync `get_token()`.
    """
    provider = AsyncOnlyTokenProvider(account_id=None)
    model = _build_model(token_provider=provider, originator=None)
    kwargs_seen: dict[str, Any] = {}

    async def _fake_super_astream(*_a: Any, **kwargs: Any) -> Any:
        kwargs_seen.update(kwargs)
        yield "chunk"

    monkeypatch.setattr(ChatOpenAI, "_astream", _fake_super_astream)
    before_sync = provider.calls
    received = [chunk async for chunk in model._astream([HumanMessage("hi")])]

    assert received == ["chunk"]
    assert provider.calls == before_sync
    assert kwargs_seen["_codex_headers"] == {}


def test_callable_api_key_returns_provider_token() -> None:
    """The `api_key` callable wired into the SDK must yield the access token."""
    provider = FakeTokenProvider(access_token="abc-123")
    model = _build_model(token_provider=provider)
    # ChatOpenAI converts callable api_keys into a `SecretStr` wrapping
    # whatever the callable returns; resolving it should return the
    # provider's current access token.
    secret = model.openai_api_key
    assert secret is not None
    if callable(secret):
        assert secret() == "abc-123"
    else:
        assert secret.get_secret_value() == "abc-123"


def test_ls_params_uses_codex_provider_tag() -> None:
    model = _build_model()
    params = model._get_ls_params()
    assert params["ls_provider"] == "openai-codex"


def test_profile_declares_no_mid_conversation_capabilities() -> None:
    """System text is flattened into one leading developer message."""
    model = _build_model(model="gpt-5.6")

    assert model.profile
    assert model.profile["mid_conversation_system_messages"] is False
    assert model.profile["mid_conversation_tools"] is False


def test_is_not_serializable_due_to_live_token_provider() -> None:
    assert _ChatOpenAICodex.is_lc_serializable() is False


def test_sync_token_callable_delegates() -> None:
    provider = FakeTokenProvider(access_token="zzz")
    callable_ = _SyncTokenCallable(provider)
    assert callable_() == "zzz"

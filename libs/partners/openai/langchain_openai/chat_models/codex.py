"""`ChatOpenAICodex`: OAuth-backed chat model for ChatGPT subscription auth.

Wraps `ChatOpenAI` to target the ChatGPT codex backend
(`https://chatgpt.com/backend-api/codex`) and supplies refresh-aware
`Authorization` and `ChatGPT-Account-Id` headers from a
`ChatGPTOAuthTokenProvider`.

The standard `ChatOpenAI` (API-key) flow is untouched.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import LangSmithParams
from pydantic import Field, PrivateAttr, model_validator

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.chatgpt_oauth import (
    ChatGPTOAuthTokenProvider,
    FileChatGPTOAuthTokenProvider,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, ChatResult
    from typing_extensions import Self


logger = logging.getLogger(__name__)


CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
ORIGINATOR_HEADER = "originator"
ORIGINATOR_VALUE = "langchain"
"""Built-in default for the `originator` header value.

Identifies requests as coming from `langchain-openai`. Override per-instance
via the `originator` field or globally via the `LANGCHAIN_CODEX_ORIGINATOR`
env var.
"""
ORIGINATOR_ENV_VAR = "LANGCHAIN_CODEX_ORIGINATOR"
ACCOUNT_ID_HEADER = "ChatGPT-Account-Id"


def _default_originator() -> str:
    """Resolve the `originator` header default, honoring the env-var override."""
    return os.environ.get(ORIGINATOR_ENV_VAR) or ORIGINATOR_VALUE


DEFAULT_INSTRUCTIONS = "You are ChatGPT, a large language model trained by OpenAI."
"""Generic fallback for the Responses-API `instructions` field.

The Codex backend rejects any request missing a top-level `instructions`
value (400 `Instructions are required`), so this constant keeps zero-config
construction working. **Most callers should override it** with their own
prompt — see `ChatOpenAICodex.instructions` for the resolution rules.
"""
_FORCED_VALUES: dict[str, Any] = {
    "use_responses_api": True,
    "output_version": "responses/v1",
    "store": False,
    "streaming": True,
}
"""Values forced onto every `ChatOpenAICodex` instance.

The ChatGPT Codex backend rejects non-streaming requests
(`400 'Stream must be set to true'`) and requests that ask it to persist
response data (`400 'Store must be set to false'`). Pinning `streaming=True`
routes `invoke` through `_stream` so a streaming request is always sent and
chunks are aggregated back into a single message for the caller.
"""


class ChatOpenAICodex(ChatOpenAI):
    """`ChatOpenAI` variant authed by a ChatGPT OAuth subscription.

    Routes requests to `https://chatgpt.com/backend-api/codex` and enforces
    Responses API behavior (`use_responses_api=True`,
    `output_version="responses/v1"`). These values are *forced* — passing a
    conflicting value to the constructor raises. Authorization and
    `ChatGPT-Account-Id` headers are taken from `token_provider` on every
    request so a freshly-refreshed access token is always used.

    Example:
        ```python
        from langchain_openai import ChatOpenAICodex
        from langchain_openai.chatgpt_oauth import login_chatgpt

        # One-time setup. The returned provider writes to the default store
        # at `~/.langchain/chatgpt-auth.json`, which `ChatOpenAICodex` also
        # reads from by default — so subsequent constructions need no
        # explicit `token_provider`.
        login_chatgpt()
        model = ChatOpenAICodex(
            model="gpt-5.5",
            instructions="You are a senior Python reviewer. Be terse.",
        )
        response = model.invoke("hello")
        ```

    !!! tip "Override `instructions`"

        The Codex backend requires a top-level `instructions` value on every
        request. A generic default keeps zero-config use working, but most
        callers should override it via the constructor (above) or per call
        (`model.invoke(..., instructions=...)`). See the field's docstring
        for the full resolution rules.

    !!! note

        Token storage is handled by `FileChatGPTOAuthTokenProvider`, which
        defaults to `~/.langchain/chatgpt-auth.json` so it does not collide
        with the Codex CLI / VS Code session at `~/.codex/auth.json`.

    !!! note "Always streams over the wire"

        The Codex backend only accepts streaming requests, so `streaming=True`
        is forced. `invoke` still returns a single aggregated `AIMessage` —
        chunks are collected internally — but the underlying HTTP request is
        a stream either way. Expect every call to show up as a streamed
        request in network logs and LangSmith traces.
    """

    token_provider: Any = Field(default=None, exclude=True)
    """Refresh-aware ChatGPT OAuth token provider.

    Must implement the `ChatGPTOAuthTokenProvider` protocol. If `None`, a
    `FileChatGPTOAuthTokenProvider` rooted at the default store path is
    constructed.
    """

    include_originator_header: bool = True
    """Whether to send the optional `originator` header."""

    originator: str = Field(default_factory=_default_originator)
    """Value sent in the `originator` request header.

    Identifies the client making the request. Defaults to `"langchain"` so
    OpenAI telemetry attributes calls to this package. Downstream consumers
    (e.g., a framework built on top of `ChatOpenAICodex`) can override this
    to identify themselves instead.

    Resolution order (first match wins):

    1. Constructor / kwarg value (`ChatOpenAICodex(originator="my-app")`).
    2. The `LANGCHAIN_CODEX_ORIGINATOR` env var, if set and non-empty.
    3. `ORIGINATOR_VALUE` (`"langchain"`).

    Ignored when `include_originator_header=False`. A per-call
    `extra_headers={"originator": "..."}` still wins over this field for
    that one call (caller-supplied headers always override Codex defaults).
    """

    instructions: str = Field(default=DEFAULT_INSTRUCTIONS)
    """System prompt sent in the Responses-API `instructions` field.

    `instructions` is a *top-level* field of the Responses API request — it
    is not a chat message. The Codex backend rejects any request where this
    field is missing or empty (400 `Instructions are required`), so this
    field always carries a value:

    - this constructor field (defaults to a generic ChatGPT prompt), or
    - an explicit `instructions=` kwarg passed to `invoke` / `stream`,
        which overrides the constructor value for that one call only.

    The Codex backend is stateless for this client (`store=False` is
    forced), so `instructions` is sent on every request and can be changed
    between calls — useful for switching persona / tooling mid-conversation:

    ```python
    model = ChatOpenAICodex(
        model="gpt-5.5",
        instructions="You are a senior Python reviewer. Be terse.",
    )
    model.invoke("review this diff…")
    model.invoke(
        "now translate the review to French",
        instructions="You are a translator.",
    )
    ```

    `instructions` is distinct from any `SystemMessage` placed in the input
    list. A `SystemMessage` becomes an entry in the `input` array (a chat
    turn); `instructions` is a separate top-level directive applied to the
    response generation. Passing a `SystemMessage` does **not** populate
    this field.
    """

    _resolved_token_provider: ChatGPTOAuthTokenProvider = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def _apply_codex_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Apply Codex-specific defaults before the parent validator runs."""
        if not isinstance(values, dict):
            return values
        for key, forced in _FORCED_VALUES.items():
            supplied = values.get(key)
            if supplied is not None and supplied != forced:
                msg = (
                    f"`ChatOpenAICodex` requires `{key}={forced!r}`; "
                    f"got `{key}={supplied!r}`. Use `ChatOpenAI` if you "
                    "need to customize this."
                )
                raise ValueError(msg)
            values[key] = forced
        values.setdefault("base_url", CHATGPT_CODEX_BASE_URL)
        # `openai_api_base` is the legacy alias for `base_url` on `ChatOpenAI`;
        # mirror whichever the caller supplied so older code paths still hit
        # the Codex endpoint.
        values.setdefault("openai_api_base", values.get("base_url"))

        provider = values.get("token_provider")
        if provider is None:
            provider = FileChatGPTOAuthTokenProvider.from_default_store()
            values["token_provider"] = provider
        if not isinstance(provider, ChatGPTOAuthTokenProvider):
            msg = (
                "`token_provider` must implement the "
                "`ChatGPTOAuthTokenProvider` protocol."
            )
            raise TypeError(msg)

        values.setdefault("api_key", _SyncTokenCallable(provider))
        return values

    @model_validator(mode="after")
    def _capture_token_provider(self) -> Self:
        """Stash the provider on a private attr for header construction."""
        self._resolved_token_provider = self.token_provider
        return self

    def _codex_headers_sync(self) -> dict[str, str]:
        token = self._resolved_token_provider.get_token()
        return self._build_headers(token.account_id)

    async def _codex_headers_async(self) -> dict[str, str]:
        token = await self._resolved_token_provider.aget_token()
        return self._build_headers(token.account_id)

    def _build_headers(self, account_id: str | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if account_id:
            headers[ACCOUNT_ID_HEADER] = account_id
        if self.include_originator_header:
            headers[ORIGINATOR_HEADER] = self.originator
        return headers

    def _merge_codex_headers(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        # Caller-supplied `extra_headers` win over our Codex defaults so
        # users can override (e.g., to send a different `originator`).
        if not headers:
            return payload
        merged = {**headers, **(payload.get("extra_headers") or {})}
        payload["extra_headers"] = merged
        return payload

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Build the request payload and attach Codex auth headers."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # The Codex backend rejects requests without `instructions` — populate
        # the field's value if the caller didn't supply one. An explicit empty
        # string from the caller is preserved (the backend will reject it, but
        # silently overwriting it would hide a programming error).
        if payload.get("instructions") is None:
            payload["instructions"] = self.instructions
        return self._merge_codex_headers(payload, self._codex_headers_sync())

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Prime the cache via async refresh so the sync header build that
        # happens inside `super()._agenerate` does not block the event loop.
        await self._resolved_token_provider.aget_token()
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        await self._resolved_token_provider.aget_token()
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "openai-codex"
        return params

    @property
    def _llm_type(self) -> str:
        return "openai-codex-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """`ChatOpenAICodex` is not serializable (holds a live token provider)."""
        return False


class _SyncTokenCallable:
    """Sync callable wrapper around a token provider for the OpenAI SDK.

    The OpenAI Python SDK accepts a callable returning a string for `api_key`.
    Wrapping the provider lets the SDK fetch a freshly-refreshed access token
    on every request without exposing the provider's other methods.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: ChatGPTOAuthTokenProvider) -> None:
        self._provider = provider

    def __call__(self) -> str:
        return self._provider.get_access_token()


__all__ = ["ChatOpenAICodex"]

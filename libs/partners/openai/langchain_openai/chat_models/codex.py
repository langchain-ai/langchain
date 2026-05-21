"""`ChatOpenAICodex`: OAuth-backed chat model for ChatGPT subscription auth.

Wraps `ChatOpenAI` to target the ChatGPT codex backend
(`https://chatgpt.com/backend-api/codex`) and supplies refresh-aware
`Authorization` and `ChatGPT-Account-Id` headers from a
`ChatGPTOAuthTokenProvider`.

The standard `ChatOpenAI` (API-key) flow is untouched.
"""

from __future__ import annotations

import logging
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
ACCOUNT_ID_HEADER = "ChatGPT-Account-Id"
_FORCED_VALUES: dict[str, Any] = {
    "use_responses_api": True,
    "output_version": "responses/v1",
}


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
        model = ChatOpenAICodex(model="gpt-5.2-codex")
        response = model.invoke("hello")
        ```

    !!! note
        Token storage is handled by `FileChatGPTOAuthTokenProvider`, which
        defaults to `~/.langchain/chatgpt-auth.json` so it does not collide
        with the Codex CLI / VS Code session at `~/.codex/auth.json`.
    """

    token_provider: Any = Field(default=None, exclude=True)
    """Refresh-aware ChatGPT OAuth token provider.

    Must implement the `ChatGPTOAuthTokenProvider` protocol. If `None`, a
    `FileChatGPTOAuthTokenProvider` rooted at the default store path is
    constructed.
    """

    include_originator_header: bool = True
    """Whether to send the optional `originator: langchain` header."""

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
            headers[ORIGINATOR_HEADER] = ORIGINATOR_VALUE
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

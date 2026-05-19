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
    from langchain_core.language_models import LanguageModelInput
    from typing_extensions import Self


logger = logging.getLogger(__name__)


CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
ORIGINATOR_HEADER = "originator"
ORIGINATOR_VALUE = "langchain"
ACCOUNT_ID_HEADER = "ChatGPT-Account-Id"


class ChatOpenAICodex(ChatOpenAI):
    """`ChatOpenAI` variant authed by a ChatGPT OAuth subscription.

    Routes requests to `https://chatgpt.com/backend-api/codex` and forces
    Responses API behavior (`use_responses_api=True`,
    `output_version="responses/v1"`). Authorization and
    `ChatGPT-Account-Id` headers are taken from `token_provider` on every
    request so a freshly-refreshed access token is always used.

    Example:
        ```python
        from langchain_openai import ChatOpenAICodex
        from langchain_openai.chatgpt_oauth import login_chatgpt

        login_chatgpt()  # one-time setup; opens browser
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
        """Force Codex-specific defaults before the parent validator runs."""
        if not isinstance(values, dict):
            return values
        values.setdefault("base_url", CHATGPT_CODEX_BASE_URL)
        values.setdefault("openai_api_base", values.get("base_url"))
        values.setdefault("use_responses_api", True)
        values.setdefault("output_version", "responses/v1")

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

    def _account_headers(
        self, token_provider: ChatGPTOAuthTokenProvider
    ) -> dict[str, str]:
        token = token_provider.get_token()
        headers: dict[str, str] = {}
        if token.account_id:
            headers[ACCOUNT_ID_HEADER] = token.account_id
        if self.include_originator_header:
            headers[ORIGINATOR_HEADER] = ORIGINATOR_VALUE
        return headers

    def _inject_codex_headers(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = self._account_headers(self._resolved_token_provider)
        if headers:
            existing = dict(payload.get("extra_headers") or {})
            existing.update(headers)
            payload["extra_headers"] = existing
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
        return self._inject_codex_headers(payload)

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

"""`_ChatOpenAICodex`: experimental OAuth-backed chat model.

Wraps `ChatOpenAI` to target the ChatGPT codex backend
(`https://chatgpt.com/backend-api/codex`) and supplies refresh-aware
`Authorization` and `ChatGPT-Account-Id` headers from a
`_ChatGPTOAuthTokenProvider`.

The standard `ChatOpenAI` (API-key) flow is untouched.

!!! warning "Experimental and unofficial"

    `_ChatOpenAICodex` is not an official OpenAI API integration. Use it only
    where your OpenAI account, workspace, plan, and applicable OpenAI terms
    permit ChatGPT-authenticated Codex access. You are responsible for ensuring
    your implementation complies with OpenAI's terms, usage policies, account
    restrictions, rate limits, and safeguards.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import BaseMessage, ChatMessage, SystemMessage
from pydantic import Field, model_validator

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai.chatgpt_oauth import (
    _ChatGPTOAuthTokenProvider,
    _FileChatGPTOAuthTokenProvider,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.outputs import ChatGenerationChunk, ChatResult


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
EXPERIMENTAL_UNOFFICIAL_WARNING = (
    "`_ChatOpenAICodex` is experimental and unofficial. It uses ChatGPT "
    "subscription OAuth against Codex endpoints and must only be used where "
    "permitted by your OpenAI account, workspace, plan, and applicable OpenAI "
    "terms and policies. You are responsible for implementing and operating "
    "it responsibly, including respecting OpenAI's usage policies, rate "
    "limits, and safeguards."
)
_experimental_warning_emitted = False
_INSTRUCTION_ROLES = frozenset({"system", "developer"})


def _default_originator() -> str:
    """Resolve the `originator` header default, honoring the env-var override."""
    return os.environ.get(ORIGINATOR_ENV_VAR) or ORIGINATOR_VALUE


def _warn_experimental_unofficial() -> None:
    """Warn once that `_ChatOpenAICodex` is experimental and unofficial."""
    global _experimental_warning_emitted
    if _experimental_warning_emitted:
        return
    _experimental_warning_emitted = True
    warnings.warn(EXPERIMENTAL_UNOFFICIAL_WARNING, UserWarning, stacklevel=5)


def _maybe_has_system_messages(input_: Any) -> bool:
    """Return `True` if `input_` *could* contain a system-role message.

    Cheap structural probe used to skip the full `_convert_input` pipeline
    when there is no chance the lift logic will fire. False positives only
    cost an extra conversion; false negatives would silently skip the lift,
    so the probe is biased toward `True` for unknown shapes.
    """
    if isinstance(input_, str):
        return False
    if isinstance(input_, BaseMessage):
        return _is_instruction_message(input_)
    if isinstance(input_, (list, tuple)):
        for item in input_:
            if isinstance(item, BaseMessage) and _is_instruction_message(item):
                return True
            if isinstance(item, dict) and item.get("role") in _INSTRUCTION_ROLES:
                return True
            if (
                isinstance(item, tuple)
                and item
                and isinstance(item[0], str)
                and item[0] in _INSTRUCTION_ROLES
            ):
                return True
        return False
    # `PromptValue` or any future shape — be safe and run the slow path.
    return True


def _is_instruction_message(message: BaseMessage) -> bool:
    return isinstance(message, SystemMessage) or (
        isinstance(message, ChatMessage) and message.role in _INSTRUCTION_ROLES
    )


def _flatten_system_message_content(system_messages: list[BaseMessage]) -> str:
    """Join system/developer message content into a single `instructions` string.

    Codex rejects system-role entries in the input list, so their content
    is lifted into the top-level `instructions` field. Content that uses
    list-of-content-blocks form is accepted only when every block is
    `{"type": "text", ...}`; anything else cannot be flattened into the
    string-typed `instructions` field.

    Raises:
        ValueError: A system/developer message carries a non-text content block.
    """
    parts: list[str] = []
    for index, message in enumerate(system_messages):
        message_name = type(message).__name__
        content = message.content
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            msg = (
                f"`{message_name}` at index {index} has unsupported content "
                f"type {type(content).__name__!r}; only `str` and "
                "list-of-text-blocks are accepted by `_ChatOpenAICodex`."
            )
            raise ValueError(msg)
        text_parts: list[str] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "text":
                msg = (
                    f"`{message_name}` at index {index} contains a "
                    f"non-text content block at position {block_index} "
                    "(Codex `instructions` is a string field — only "
                    '`{"type": "text", "text": "..."}` blocks can be '
                    "lifted into it). Move the non-text content to a "
                    "`HumanMessage`, or pass plain instructions via the "
                    "constructor or `instructions=` kwarg."
                )
                raise ValueError(msg)
            text_value = block.get("text", "")
            if not isinstance(text_value, str):
                msg = (
                    f"`{message_name}` at index {index} has a text block "
                    f"at position {block_index} whose `text` is not a "
                    "string."
                )
                raise ValueError(msg)
            text_parts.append(text_value)
        parts.append("".join(text_parts))
    return "\n\n".join(parts)


DEFAULT_INSTRUCTIONS = "You are ChatGPT, a large language model trained by OpenAI."
"""Generic fallback for the Responses-API `instructions` field.

The Codex backend rejects any request missing a top-level `instructions`
value (400 `Instructions are required`), so this constant keeps zero-config
construction working. **Most callers should override it** with their own
prompt — see `_ChatOpenAICodex.instructions` for the resolution rules.
"""
_FORCED_VALUES: dict[str, Any] = {
    "use_responses_api": True,
    "store": False,
    "streaming": True,
}
"""Values forced onto every `_ChatOpenAICodex` instance.

These are the wire-level constraints the Codex backend imposes:

- `use_responses_api=True`: Codex is only reachable through the Responses
    API surface.
- `store=False`: the backend rejects `store=true`
    (`400 'Store must be set to false'`).
- `streaming=True`: the backend rejects non-streaming requests
    (`400 'Stream must be set to true'`). Pinning this routes `invoke`
    through `_stream` so a streaming request is always sent and chunks
    are aggregated back into a single message for the caller.

`output_version` is intentionally **not** forced — it is a client-side
`AIMessage` projection (see `ChatOpenAI.output_version`) that never
appears in the request payload, so callers can pick `"v0"`, `"v1"`, or
`"responses/v1"` freely.

`base_url` (and its `openai_api_base` alias) is also pinned — to
`CHATGPT_CODEX_BASE_URL` — under the same raise-don't-rewrite contract.
It is enforced separately in the validator rather than listed here
because a caller-controlled endpoint combined with the OAuth bearer
token would be a token-exfiltration vector; see the validator for the
rationale.
"""


class _ChatOpenAICodex(ChatOpenAI):
    """Experimental `ChatOpenAI` variant authed by ChatGPT OAuth.

    This integration is unofficial and should only be used where your OpenAI
    account, workspace, plan, and applicable OpenAI terms permit
    ChatGPT-authenticated Codex access. Users are responsible for implementing
    and operating it in compliance with OpenAI's terms, usage policies, account
    restrictions, rate limits, and safeguards.

    Routes requests to `https://chatgpt.com/backend-api/codex` and forces
    the wire-level fields the Codex backend requires
    (`use_responses_api=True`, `store=False`, `streaming=True`). These
    values are forced — passing a conflicting value to the constructor
    raises. `output_version` (a client-side `AIMessage` projection) is
    not forced; pick whichever projection you want. Authorization and
    `ChatGPT-Account-Id` headers are taken from `token_provider` on every
    request so a freshly-refreshed access token is always used.

    Example:
        ```python
        from langchain_openai.chat_models.codex import _ChatOpenAICodex
        from langchain_openai.chatgpt_oauth import login_chatgpt

        # One-time setup. The returned provider writes to the default store
        # at `~/.langchain/chatgpt-auth.json`, which `_ChatOpenAICodex` also
        # reads from by default — so subsequent constructions need no
        # explicit `token_provider`.
        login_chatgpt()
        model = _ChatOpenAICodex(
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

        Token storage is handled by `_FileChatGPTOAuthTokenProvider`, which
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

    Must implement the `_ChatGPTOAuthTokenProvider` protocol. If `None`, a
    `_FileChatGPTOAuthTokenProvider` rooted at the default store path is
    constructed.
    """

    originator: str | None = Field(default_factory=_default_originator)
    """Value sent in the `originator` request header, or `None` to omit it.

    Identifies the client making the request. Defaults to `"langchain"` so
    OpenAI telemetry attributes calls to this package. Downstream consumers
    (e.g., a framework built on top of `_ChatOpenAICodex`) can override this
    to identify themselves instead, or set `None` to suppress the header.

    Resolution order (first match wins):

    1. Per-call `extra_headers={"originator": "..."}` (always trumps the
        field; pass an explicit value to override on a single call).
    2. Constructor / kwarg value (`_ChatOpenAICodex(originator="my-app")`).
    3. The `LANGCHAIN_CODEX_ORIGINATOR` env var, if set and non-empty.
    4. `ORIGINATOR_VALUE` (`"langchain"`).

    Setting `originator=None` disables the header entirely; the constructor
    default never resolves to `None`.
    """

    instructions: str = Field(default=DEFAULT_INSTRUCTIONS)
    """System prompt sent in the Responses-API `instructions` field.

    `instructions` is a *top-level* field of the Responses API request — it
    is not a chat message. The Codex backend rejects any request where this
    field is missing or empty (400 `Instructions are required`) **and**
    rejects any `SystemMessage` entry in the input list
    (400 `System messages are not allowed`). To bridge those constraints
    transparently, `_ChatOpenAICodex` resolves `instructions` per call with
    this precedence (highest wins):

    1. Explicit `instructions=` kwarg on `invoke` / `stream`.
    2. Concatenated content of any `SystemMessage` entries in the input
        list — joined with `"\\n\\n"` and stripped from the input before
        sending. Set the explicit kwarg in (1) to override.
    3. This constructor field (defaults to a generic ChatGPT prompt).

    The Codex backend is stateless for this client (`store=False` is
    forced), so `instructions` is sent on every request and can be changed
    between calls — useful for switching persona / tooling mid-conversation:

    ```python
    model = _ChatOpenAICodex(
        model="gpt-5.5",
        instructions="You are a senior Python reviewer. Be terse.",
    )
    model.invoke("review this diff…")
    model.invoke(
        "now translate the review to French",
        instructions="You are a translator.",
    )
    ```

    `SystemMessage` content that uses list-of-content-blocks form is
    accepted only if every block is `{"type": "text", ...}`; any other
    block type raises `ValueError` since it cannot be flattened into the
    string-typed `instructions` field.
    """

    @model_validator(mode="before")
    @classmethod
    def _apply_codex_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Apply Codex-specific defaults before the parent validator runs."""
        _warn_experimental_unofficial()
        if not isinstance(values, dict):
            return values
        for key, forced in _FORCED_VALUES.items():
            supplied = values.get(key)
            if supplied is not None and supplied != forced:
                msg = (
                    f"`_ChatOpenAICodex` requires `{key}={forced!r}`; "
                    f"got `{key}={supplied!r}`. Use `ChatOpenAI` if you "
                    "need to customize this."
                )
                raise ValueError(msg)
            values[key] = forced
        # Pin `base_url` (and its legacy `openai_api_base` alias) to the Codex
        # endpoint. The OAuth bearer token is wired in as `api_key` below, so a
        # caller-controlled `base_url` would otherwise exfiltrate the token to
        # an attacker-chosen host. Reject any non-matching override rather than
        # silently rewriting it, mirroring the `_FORCED_VALUES` contract.
        for key in ("base_url", "openai_api_base"):
            supplied = values.get(key)
            if supplied is not None and supplied != CHATGPT_CODEX_BASE_URL:
                msg = (
                    f"`_ChatOpenAICodex` requires `{key}={CHATGPT_CODEX_BASE_URL!r}`; "
                    f"got `{key}={supplied!r}`. Use `ChatOpenAI` if you need to "
                    "target a different endpoint."
                )
                raise ValueError(msg)
            values[key] = CHATGPT_CODEX_BASE_URL

        provider = values.get("token_provider")
        if provider is None:
            provider = _FileChatGPTOAuthTokenProvider.from_default_store()
            values["token_provider"] = provider
        if not isinstance(provider, _ChatGPTOAuthTokenProvider):
            msg = (
                "`token_provider` must implement the "
                "`_ChatGPTOAuthTokenProvider` protocol."
            )
            raise TypeError(msg)

        # The OAuth `token_provider` is the sole auth source: its access token
        # is wired into the OpenAI SDK as `api_key` below. A caller-supplied
        # `api_key` (or its `openai_api_key` alias) would silently win over the
        # OAuth bearer, leaving the model in a conflicting state — so reject it
        # (raise-don't-rewrite, mirroring the `base_url` handling above). An
        # `OPENAI_API_KEY` env var is not consulted: the field's default
        # factory never runs because `api_key` is always set here.
        for key in ("api_key", "openai_api_key"):
            if values.get(key) is not None:
                msg = (
                    f"`_ChatOpenAICodex` manages authentication via "
                    f"`token_provider`; drop the explicit `{key}=`. Use "
                    "`ChatOpenAI` if you want API-key authentication."
                )
                raise ValueError(msg)
        values["api_key"] = _SyncTokenCallable(provider)
        return values

    def _codex_headers_sync(self) -> dict[str, str]:
        token = self.token_provider.get_token()
        return self._build_headers(token.account_id)

    def _build_headers(self, account_id: str | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if account_id:
            headers[ACCOUNT_ID_HEADER] = account_id
        if self.originator is not None:
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
        """Build the request payload and attach Codex auth headers.

        Lifts any `SystemMessage` content out of the input list into the
        top-level `instructions` field, since Codex rejects `SystemMessage`
        chat turns. See the `instructions` field docstring for the
        precedence rules.

        Fast path: when the input can't carry a `SystemMessage`, skip the
        local conversion and delegate `input_` straight to super — that
        way `_convert_input` only runs once (inside super) instead of once
        here and again there.
        """
        codex_headers = kwargs.pop("_codex_headers", None)
        payload_input: LanguageModelInput = input_
        if _maybe_has_system_messages(input_):
            messages = self._convert_input(input_).to_messages()
            system_messages = [m for m in messages if _is_instruction_message(m)]
            if system_messages:
                non_system = [m for m in messages if not _is_instruction_message(m)]
                lifted = _flatten_system_message_content(system_messages)
                explicit = kwargs.get("instructions")
                if explicit is not None:
                    logger.warning(
                        "Both `instructions=` and a `SystemMessage` were "
                        "provided; the explicit `instructions=` kwarg wins "
                        "and the `SystemMessage` content is discarded for "
                        "this call. Discarded length: %d.",
                        len(lifted),
                    )
                else:
                    kwargs["instructions"] = lifted
                payload_input = non_system

        payload = super()._get_request_payload(payload_input, stop=stop, **kwargs)
        # The Codex backend rejects requests without `instructions` — populate
        # the field's value if the caller didn't supply one. An explicit empty
        # string from the caller is preserved (the backend will reject it, but
        # silently overwriting it would hide a programming error).
        if payload.get("instructions") is None:
            payload["instructions"] = self.instructions
        headers = (
            codex_headers if codex_headers is not None else self._codex_headers_sync()
        )
        return self._merge_codex_headers(payload, headers)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Prime the cache via async refresh so the sync header build that
        # happens inside `super()._agenerate` does not block the event loop.
        await self.token_provider.aget_token()
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
        token = await self.token_provider.aget_token()
        kwargs["_codex_headers"] = self._build_headers(token.account_id)
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
        """`_ChatOpenAICodex` is not serializable (holds a live token provider)."""
        return False


class _SyncTokenCallable:
    """Sync callable wrapper around a token provider for the OpenAI SDK.

    The OpenAI Python SDK accepts a callable returning a string for `api_key`.
    Wrapping the provider lets the SDK fetch a freshly-refreshed access token
    on every request without exposing the provider's other methods.
    """

    __slots__ = ("_provider",)

    def __init__(self, provider: _ChatGPTOAuthTokenProvider) -> None:
        self._provider = provider

    def __call__(self) -> str:
        return self._provider.get_access_token()


__all__: list[str] = []

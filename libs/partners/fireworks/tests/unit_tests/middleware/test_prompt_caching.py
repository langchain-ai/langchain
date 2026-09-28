"""Unit tests for `FireworksPromptCachingMiddleware`."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_fireworks import ChatFireworks
from langchain_fireworks.middleware import FireworksPromptCachingMiddleware
from langchain_fireworks.middleware.prompt_caching import _SESSION_AFFINITY_HEADER

_THREAD_ID = "thread-abc-123"
_MODEL_NAME = "accounts/fireworks/models/test-model"


def _make_model(**kwargs: Any) -> ChatFireworks:
    settings: dict[str, Any] = {
        "model": _MODEL_NAME,
        "api_key": "fake-key",
        **kwargs,
    }
    return ChatFireworks(**settings)


def _make_request(
    model: ChatFireworks | GenericFakeChatModel,
    model_settings: dict[str, Any] | None = None,
) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=[],
        model_settings=model_settings if model_settings is not None else {},
    )


def _run(
    request: ModelRequest,
    *,
    middleware: FireworksPromptCachingMiddleware | None = None,
    thread_id: str | None = _THREAD_ID,
    config: dict[str, Any] | None = None,
) -> ModelRequest:
    middleware = middleware or FireworksPromptCachingMiddleware()
    captured: dict[str, ModelRequest] = {}

    def handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        return ModelResponse(result=[AIMessage(content="ok")])

    if config is None:
        config = (
            {"configurable": {"thread_id": thread_id}} if thread_id is not None else {}
        )
    with patch(
        "langchain_fireworks.middleware.prompt_caching.get_config",
        return_value=config,
    ):
        middleware.wrap_model_call(request, handler)
    return captured["request"]


async def _arun(
    request: ModelRequest,
    *,
    middleware: FireworksPromptCachingMiddleware | None = None,
    config: dict[str, Any] | None = None,
) -> ModelRequest:
    middleware = middleware or FireworksPromptCachingMiddleware()
    captured: dict[str, ModelRequest] = {}

    async def handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        return ModelResponse(result=[AIMessage(content="ok")])

    if config is None:
        config = {"configurable": {"thread_id": _THREAD_ID}}
    with patch(
        "langchain_fireworks.middleware.prompt_caching.get_config",
        return_value=config,
    ):
        await middleware.awrap_model_call(request, handler)
    return captured["request"]


def test_fireworks_model_injects_session_affinity() -> None:
    request = _make_request(_make_model())
    result = _run(request)

    assert result.model_settings["prompt_cache_key"] == _THREAD_ID
    assert (
        result.model_settings["extra_headers"][_SESSION_AFFINITY_HEADER] == _THREAD_ID
    )


def test_unsupported_model_behavior() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    request = _make_request(model)

    ignored = _run(
        request,
        middleware=FireworksPromptCachingMiddleware(
            unsupported_model_behavior="ignore"
        ),
    )
    assert ignored is request

    with pytest.warns(UserWarning, match="only supports ChatFireworks"):
        warned = _run(request)
    assert warned is request

    with pytest.raises(ValueError, match="only supports ChatFireworks"):
        _run(
            request,
            middleware=FireworksPromptCachingMiddleware(
                unsupported_model_behavior="raise"
            ),
        )


def test_invalid_unsupported_model_behavior_raises() -> None:
    with pytest.raises(ValueError, match="unsupported_model_behavior must be one of"):
        FireworksPromptCachingMiddleware(
            unsupported_model_behavior="warning",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("config", "thread_id"),
    [
        ({}, None),
        ({"configurable": None}, _THREAD_ID),
        (None, ""),
        ({"configurable": {"thread_id": 123}}, None),
    ],
)
def test_missing_thread_id_is_unchanged(
    config: dict[str, Any] | None,
    thread_id: str | None,
) -> None:
    request = _make_request(_make_model())
    result = _run(request, config=config, thread_id=thread_id)

    assert result is request
    assert result.model_settings == {}


def test_no_runnable_context_is_unchanged() -> None:
    request = _make_request(_make_model())
    middleware = FireworksPromptCachingMiddleware()
    captured: dict[str, ModelRequest] = {}

    def handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        return ModelResponse(result=[AIMessage(content="ok")])

    with patch(
        "langchain_fireworks.middleware.prompt_caching.get_config",
        side_effect=RuntimeError,
    ):
        middleware.wrap_model_call(request, handler)

    assert captured["request"] is request


@pytest.mark.parametrize("setting", ["user", "prompt_cache_key"])
def test_existing_affinity_setting_causes_no_injection(setting: str) -> None:
    request = _make_request(_make_model(), {setting: "caller"})
    result = _run(request)

    assert result is request
    assert result.model_settings == {setting: "caller"}


@pytest.mark.parametrize("setting", ["user", "prompt_cache_key"])
def test_model_affinity_setting_causes_no_injection(setting: str) -> None:
    request = _make_request(_make_model(model_kwargs={setting: "caller"}))
    result = _run(request)

    assert result is request
    assert result.model_settings == {}


@pytest.mark.parametrize("setting", ["user", "prompt_cache_key"])
def test_null_affinity_setting_injects_session_affinity(setting: str) -> None:
    request = _make_request(_make_model(), {setting: None})
    result = _run(request)

    if setting == "user":
        assert result.model_settings[setting] is None
    assert result.model_settings["prompt_cache_key"] == _THREAD_ID
    assert (
        result.model_settings["extra_headers"][_SESSION_AFFINITY_HEADER] == _THREAD_ID
    )


def test_existing_session_affinity_header_causes_no_injection() -> None:
    request = _make_request(
        _make_model(),
        {"extra_headers": {"X-Session-Affinity": "existing"}},
    )
    result = _run(request)

    assert result is request
    assert result.model_settings["extra_headers"] == {"X-Session-Affinity": "existing"}


def test_model_headers_stay_on_model_without_mutation() -> None:
    model_headers = {"X-Model-Header": "model-value"}
    request_headers = {"X-Request-ID": "request-1"}
    model = _make_model(model_kwargs={"extra_headers": model_headers})
    request = _make_request(model, {"extra_headers": request_headers})

    result = _run(request)

    assert result.model_settings["extra_headers"] == {
        "X-Request-ID": "request-1",
        _SESSION_AFFINITY_HEADER: _THREAD_ID,
    }
    assert model_headers == {"X-Model-Header": "model-value"}
    assert request_headers == {"X-Request-ID": "request-1"}


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("fireworks_fallback", [False, True])
async def test_model_headers_do_not_leak_to_fallback(
    *, use_async: bool, fireworks_fallback: bool
) -> None:
    primary_headers = {"Authorization": "primary-placeholder"}
    primary = _make_model(model_kwargs={"extra_headers": primary_headers})
    primary.client = MagicMock()
    primary.client.create.side_effect = ValueError("primary failed")
    primary.async_client = MagicMock()
    primary.async_client.create = AsyncMock(side_effect=ValueError("primary failed"))
    fallback: ChatFireworks | GenericFakeChatModel
    if fireworks_fallback:
        fallback = _make_model(
            model_kwargs={"extra_headers": {"Authorization": "fallback-placeholder"}}
        )
        response = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        fallback.client = MagicMock()
        fallback.client.create.return_value = response
        fallback.async_client = MagicMock()
        fallback.async_client.create = AsyncMock(return_value=response)
    else:
        fallback = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))

    request_headers = {"X-Request-ID": "request-1"}
    request = _make_request(primary, {"extra_headers": request_headers})
    caching = FireworksPromptCachingMiddleware()
    fallbacks = ModelFallbackMiddleware(fallback)
    attempts: list[ModelRequest] = []

    def handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        return ModelResponse(result=[req.model.invoke("Hello", **req.model_settings)])

    async def ahandler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        message = await req.model.ainvoke("Hello", **req.model_settings)
        return ModelResponse(result=[message])

    with patch(
        "langchain_fireworks.middleware.prompt_caching.get_config",
        return_value={"configurable": {"thread_id": _THREAD_ID}},
    ):
        if use_async:

            async def afallback(req: ModelRequest) -> ModelResponse:
                result = await fallbacks.awrap_model_call(req, ahandler)
                assert isinstance(result, ModelResponse)
                return result

            await caching.awrap_model_call(request, afallback)
        else:

            def fallback_handler(req: ModelRequest) -> ModelResponse:
                result = fallbacks.wrap_model_call(req, handler)
                assert isinstance(result, ModelResponse)
                return result

            caching.wrap_model_call(request, fallback_handler)

    assert len(attempts) == 2
    for attempt in attempts:
        assert "Authorization" not in attempt.model_settings["extra_headers"]
        assert attempt.model_settings["extra_headers"]["X-Request-ID"] == "request-1"
    client = primary.async_client if use_async else primary.client
    assert client.create.call_args.kwargs["extra_headers"] == {
        **primary_headers,
        **request_headers,
        _SESSION_AFFINITY_HEADER: _THREAD_ID,
    }
    if isinstance(fallback, ChatFireworks):
        client = fallback.async_client if use_async else fallback.client
        assert client.create.call_args.kwargs["extra_headers"] == {
            "Authorization": "fallback-placeholder",
            **request_headers,
            _SESSION_AFFINITY_HEADER: _THREAD_ID,
        }
    assert primary.model_kwargs["extra_headers"] == primary_headers
    assert request.model_settings == {"extra_headers": request_headers}


def test_conflicting_header_prefers_request_value() -> None:
    model = _make_model(model_kwargs={"extra_headers": {"X-Shared": "model"}})
    request = _make_request(model, {"extra_headers": {"X-Shared": "request"}})

    result = _run(request)

    assert result.model_settings["extra_headers"]["X-Shared"] == "request"
    assert (
        result.model_settings["extra_headers"][_SESSION_AFFINITY_HEADER] == _THREAD_ID
    )


def test_non_mapping_extra_headers_is_unchanged_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = _make_request(
        _make_model(),
        {"extra_headers": ["not", "a", "mapping"]},
    )

    with caplog.at_level(
        logging.WARNING,
        logger="langchain_fireworks.middleware.prompt_caching",
    ):
        result = _run(request)

    assert result is request
    assert any("extra_headers" in record.message for record in caplog.records)


def test_thread_id_is_not_logged() -> None:
    request = _make_request(_make_model())

    with patch("langchain_fireworks.middleware.prompt_caching.logger") as mock_logger:
        _run(request)

    calls = mock_logger.debug.call_args_list + mock_logger.warning.call_args_list
    logged = " ".join(str(arg) for call in calls for arg in call.args)
    assert _THREAD_ID not in logged


async def test_async_fireworks_model_injects_session_affinity() -> None:
    request = _make_request(_make_model())
    result = await _arun(request)

    assert result.model_settings["prompt_cache_key"] == _THREAD_ID
    assert (
        result.model_settings["extra_headers"][_SESSION_AFFINITY_HEADER] == _THREAD_ID
    )


async def test_async_missing_thread_id_passes_original_request() -> None:
    # No thread_id -> `_apply_session_affinity` returns None; the async path must
    # fall back to the original request rather than pass `None` to the handler.
    request = _make_request(_make_model())
    result = await _arun(request, config={})

    assert result is request
    assert result.model_settings == {}


async def test_async_unsupported_model_passes_original_request() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    request = _make_request(model)
    result = await _arun(
        request,
        middleware=FireworksPromptCachingMiddleware(
            unsupported_model_behavior="ignore"
        ),
    )

    assert result is request

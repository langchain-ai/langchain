"""Unit tests for `FireworksPromptCachingMiddleware`."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
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


def test_headers_are_merged_without_mutation() -> None:
    model_headers = {"X-Model-Header": "model-value"}
    request_headers = {"X-Request-ID": "request-1"}
    model = _make_model(model_kwargs={"extra_headers": model_headers})
    request = _make_request(model, {"extra_headers": request_headers})

    result = _run(request)

    assert result.model_settings["extra_headers"] == {
        "X-Model-Header": "model-value",
        "X-Request-ID": "request-1",
        _SESSION_AFFINITY_HEADER: _THREAD_ID,
    }
    assert model_headers == {"X-Model-Header": "model-value"}
    assert request_headers == {"X-Request-ID": "request-1"}


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

"""Unit tests for `FireworksPromptCachingMiddleware`."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fireworks import AsyncFireworks, Fireworks
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, wrap_model_call
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from langchain_fireworks import ChatFireworks
from langchain_fireworks.middleware import FireworksPromptCachingMiddleware

_THREAD_ID = "thread-abc-123"
_AFFINITY = "ef8dbcc47038744819e7c6386305c0acf1961597773391096c267358ec6496eb"
_MODEL_NAME = "accounts/fireworks/models/test-model"
_SESSION_AFFINITY_HEADER = "x-session-affinity"


def _make_model(**kwargs: Any) -> ChatFireworks:
    settings: dict[str, Any] = {
        "model": _MODEL_NAME,
        "api_key": "fake-key",
        **kwargs,
    }
    model = ChatFireworks(**settings)
    response = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    model.client = MagicMock()
    model.client.create.return_value = response
    model.async_client = MagicMock()
    model.async_client.create = AsyncMock(return_value=response)
    return model


def _call_kwargs(request: ModelRequest) -> dict[str, Any]:
    assert isinstance(request.model, ChatFireworks)
    return dict(request.model.client.create.call_args.kwargs)


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
        return ModelResponse(result=[req.model.invoke("Hello", **req.model_settings)])

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
) -> ModelRequest:
    middleware = middleware or FireworksPromptCachingMiddleware()
    captured: dict[str, ModelRequest] = {}

    async def handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        message = await req.model.ainvoke("Hello", **req.model_settings)
        return ModelResponse(result=[message])

    with patch(
        "langchain_fireworks.middleware.prompt_caching.get_config",
        return_value={"configurable": {"thread_id": _THREAD_ID}},
    ):
        await middleware.awrap_model_call(request, handler)
    return captured["request"]


@pytest.mark.parametrize(
    "thread_id",
    [
        "会話-café-😀",
        "thread\r\nInjected: value\x00",
        "x" * 1024,
    ],
    ids=["unicode", "control-characters", "long"],
)
def test_thread_id_produces_stable_header_safe_affinity(thread_id: str) -> None:
    request = _make_request(_make_model())

    def run(value: str) -> dict[str, Any]:
        _run(request, thread_id=value)
        return _call_kwargs(request)

    result = run(thread_id)
    affinity = httpx.Headers(result["extra_headers"])[_SESSION_AFFINITY_HEADER]
    assert result["prompt_cache_key"] == affinity
    assert len(affinity) == 64
    assert set(affinity) <= set("0123456789abcdef")
    assert run(thread_id)["prompt_cache_key"] == affinity
    assert run(thread_id + "-other")["prompt_cache_key"] != affinity
    assert request.model_settings == {}


def test_unsupported_model_behavior() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 2))
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
    assert "prompt_cache_key" not in _call_kwargs(request)
    assert "extra_headers" not in _call_kwargs(request)


def test_no_runnable_context_is_unchanged() -> None:
    request = _make_request(_make_model())
    middleware = FireworksPromptCachingMiddleware()
    captured: dict[str, ModelRequest] = {}

    def handler(req: ModelRequest) -> ModelResponse:
        captured["request"] = req
        return ModelResponse(result=[req.model.invoke("Hello", **req.model_settings)])

    with patch(
        "langchain_fireworks.middleware.prompt_caching.get_config",
        side_effect=RuntimeError,
    ):
        middleware.wrap_model_call(request, handler)

    assert captured["request"] is request
    assert "prompt_cache_key" not in _call_kwargs(request)
    assert "extra_headers" not in _call_kwargs(request)


@pytest.mark.parametrize("setting", ["user", "prompt_cache_key"])
def test_existing_affinity_setting_causes_no_injection(setting: str) -> None:
    request = _make_request(_make_model(), {setting: "caller"})
    result = _run(request)

    assert result is request
    assert result.model_settings == {setting: "caller"}
    assert _call_kwargs(request)[setting] == "caller"
    assert "extra_headers" not in _call_kwargs(request)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("on_model", [False, True])
@pytest.mark.parametrize("setting", ["user", "prompt_cache_key"])
async def test_extra_body_affinity_takes_precedence_on_the_wire(
    setting: str, *, use_async: bool, on_model: bool
) -> None:
    requests: list[httpx.Request] = []
    extra_body = {setting: "explicit-affinity"}

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    with Fireworks(
        api_key="fake-key",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as sdk:
        async with AsyncFireworks(
            api_key="fake-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as async_sdk:
            model = ChatFireworks(
                model=_MODEL_NAME,
                api_key="fake-key",  # type: ignore[arg-type]
                client=sdk.chat.completions,
                async_client=async_sdk.chat.completions,
                model_kwargs={"extra_body": extra_body} if on_model else {},
            )
            request = _make_request(
                model, {} if on_model else {"extra_body": extra_body}
            )
            if use_async:
                await _arun(request)
            else:
                _run(request)

    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert body[setting] == "explicit-affinity"
    assert body.get("prompt_cache_key") == extra_body.get("prompt_cache_key")
    assert _SESSION_AFFINITY_HEADER not in requests[0].headers
    assert extra_body == {setting: "explicit-affinity"}
    assert model.model_kwargs == ({"extra_body": extra_body} if on_model else {})
    assert request.model_settings == ({} if on_model else {"extra_body": extra_body})


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("setting", ["user", "prompt_cache_key"])
async def test_extra_body_null_overrides_top_level_affinity(
    setting: str, *, use_async: bool
) -> None:
    settings = {setting: "overridden", "extra_body": {setting: None}}
    model = _make_model()
    request = _make_request(model, settings)
    if use_async:
        await _arun(request)
    else:
        _run(request)

    client = model.async_client if use_async else model.client
    kwargs = client.create.call_args.kwargs
    assert kwargs["extra_headers"][_SESSION_AFFINITY_HEADER] == _AFFINITY
    assert kwargs["prompt_cache_key"] == _AFFINITY
    assert kwargs["extra_body"] == {setting: None}
    assert settings == {setting: "overridden", "extra_body": {setting: None}}


@pytest.mark.parametrize("setting", ["user", "prompt_cache_key"])
def test_null_affinity_setting_injects_session_affinity(setting: str) -> None:
    request = _make_request(_make_model(), {setting: None})
    result = _run(request)

    assert result is request
    kwargs = _call_kwargs(request)
    if setting == "user":
        assert kwargs[setting] is None
    assert kwargs["prompt_cache_key"] == _AFFINITY
    assert kwargs["extra_headers"][_SESSION_AFFINITY_HEADER] == _AFFINITY


def test_existing_session_affinity_header_causes_no_injection() -> None:
    request = _make_request(
        _make_model(),
        {"extra_headers": {"X-Session-Affinity": "existing"}},
    )
    result = _run(request)

    assert result is request
    assert result.model_settings["extra_headers"] == {"X-Session-Affinity": "existing"}
    assert _call_kwargs(request)["extra_headers"] == {"X-Session-Affinity": "existing"}
    assert "prompt_cache_key" not in _call_kwargs(request)


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
        assert attempt.model_settings == {"extra_headers": request_headers}
        assert "Authorization" not in attempt.model_settings["extra_headers"]
        assert attempt.model_settings["extra_headers"]["X-Request-ID"] == "request-1"
    client = primary.async_client if use_async else primary.client
    assert client.create.call_args.kwargs["extra_headers"] == {
        **primary_headers,
        **request_headers,
        _SESSION_AFFINITY_HEADER: _AFFINITY,
    }
    if isinstance(fallback, ChatFireworks):
        client = fallback.async_client if use_async else fallback.client
        assert client.create.call_args.kwargs["extra_headers"] == {
            "Authorization": "fallback-placeholder",
            **request_headers,
            _SESSION_AFFINITY_HEADER: _AFFINITY,
        }
    assert primary.model_kwargs["extra_headers"] == primary_headers
    assert request.model_settings == {"extra_headers": request_headers}


def test_non_mapping_extra_headers_is_unchanged_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request = _make_request(
        _make_model(),
        {"extra_headers": ["not", "a", "mapping"]},
    )

    with caplog.at_level(
        logging.WARNING,
        logger="langchain_fireworks.chat_models",
    ):
        result = _run(request)

    assert result is request
    assert any("extra_headers" in record.message for record in caplog.records)


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


@pytest.mark.parametrize("use_async", [False, True])
async def test_agent_thread_id_without_runnable_context(*, use_async: bool) -> None:
    """Runtime metadata supplies affinity without ambient config on Python 3.10."""
    model = _make_model()
    agent = create_agent(model, middleware=[FireworksPromptCachingMiddleware()])
    config: RunnableConfig = {"configurable": {"thread_id": _THREAD_ID}}

    with patch(
        "langchain_fireworks.middleware.prompt_caching.get_config",
        side_effect=RuntimeError("No runnable context"),
    ):
        if use_async:
            await agent.ainvoke({"messages": [HumanMessage("Hello")]}, config)
        else:
            agent.invoke({"messages": [HumanMessage("Hello")]}, config)

    client = model.async_client if use_async else model.client
    kwargs = client.create.call_args.kwargs
    assert kwargs["prompt_cache_key"] == _AFFINITY
    assert kwargs["extra_headers"][_SESSION_AFFINITY_HEADER] == _AFFINITY


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("primary_settings", "fallback_settings", "caching_first"),
    [
        pytest.param({}, {}, False, id="automatic-fallback-first"),
        pytest.param({}, {}, True, id="automatic-caching-first"),
        pytest.param({}, {"user": "fallback-user"}, True, id="fallback-user"),
        pytest.param(
            {}, {"prompt_cache_key": "fallback-cache"}, True, id="fallback-key"
        ),
        pytest.param(
            {},
            {"extra_headers": {"X-Session-Affinity": "fallback-affinity"}},
            True,
            id="fallback-header",
        ),
        pytest.param(
            {"prompt_cache_key": "primary-cache"},
            {},
            False,
            id="primary-key-fallback-first",
        ),
        pytest.param(
            {"prompt_cache_key": "primary-cache"},
            {},
            True,
            id="primary-key-caching-first",
        ),
    ],
)
async def test_fallback_respects_explicit_affinity(
    primary_settings: dict[str, Any],
    fallback_settings: dict[str, Any],
    *,
    use_async: bool,
    caching_first: bool,
) -> None:
    primary = _make_model(model_kwargs=primary_settings)
    primary.client = MagicMock()
    primary.client.create.side_effect = ValueError("primary failed")
    primary.async_client = MagicMock()
    primary.async_client.create = AsyncMock(side_effect=ValueError("primary failed"))
    fallback = _make_model(model_kwargs=fallback_settings)
    caching = FireworksPromptCachingMiddleware()
    fallbacks = ModelFallbackMiddleware(fallback)
    agent = create_agent(
        primary,
        middleware=[caching, fallbacks] if caching_first else [fallbacks, caching],
    )
    config: RunnableConfig = {"configurable": {"thread_id": _THREAD_ID}}
    if use_async:
        await agent.ainvoke({"messages": [HumanMessage("Hello")]}, config)
    else:
        agent.invoke({"messages": [HumanMessage("Hello")]}, config)

    client = fallback.async_client if use_async else fallback.client
    kwargs = client.create.call_args.kwargs
    expected = fallback_settings or {
        "prompt_cache_key": _AFFINITY,
        "extra_headers": {_SESSION_AFFINITY_HEADER: _AFFINITY},
    }
    for setting in ("user", "prompt_cache_key", "extra_headers"):
        assert kwargs.get(setting) == expected.get(setting)
    assert fallback.model_kwargs == fallback_settings
    client = primary.async_client if use_async else primary.client
    assert client.create.call_args.kwargs["prompt_cache_key"] == primary_settings.get(
        "prompt_cache_key", _AFFINITY
    )
    assert primary.model_kwargs == primary_settings


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_affinity_is_cleared_after_handler(
    *, use_async: bool, fail: bool
) -> None:
    model = _make_model()
    client = model.async_client if use_async else model.client
    request = _make_request(model)
    if fail:
        client.create.side_effect = ValueError("model failed")
        with pytest.raises(ValueError, match="model failed"):
            if use_async:
                await _arun(request)
            else:
                _run(request)
        client.create.side_effect = None
    elif use_async:
        await _arun(request)
    else:
        _run(request)

    assert client.create.call_args.kwargs["prompt_cache_key"] == _AFFINITY
    if use_async:
        await model.ainvoke("Outside middleware")
    else:
        model.invoke("Outside middleware")
    assert "prompt_cache_key" not in client.create.call_args.kwargs
    assert "extra_headers" not in client.create.call_args.kwargs


async def test_concurrent_agents_keep_affinity_separate() -> None:
    model = _make_model()
    ready = asyncio.Event()
    arrivals = 0

    @wrap_model_call
    async def wait_for_both_calls(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            ready.set()
        await ready.wait()
        return await handler(request)

    agent = create_agent(
        model, middleware=[FireworksPromptCachingMiddleware(), wait_for_both_calls]
    )
    await asyncio.wait_for(
        asyncio.gather(
            *(
                agent.ainvoke(
                    {"messages": [HumanMessage(thread_id)]},
                    {"configurable": {"thread_id": thread_id}},
                )
                for thread_id in (_THREAD_ID, "other-thread")
            )
        ),
        timeout=10,
    )
    keys = {
        call.kwargs["messages"][0]["content"]: call.kwargs["prompt_cache_key"]
        for call in model.async_client.create.call_args_list
    }
    assert keys[_THREAD_ID] == _AFFINITY
    assert keys["other-thread"] != _AFFINITY
    await model.ainvoke("Outside middleware")
    assert "prompt_cache_key" not in model.async_client.create.call_args.kwargs

"""Unit tests for Perplexity attribution headers."""

import base64
import struct
from importlib import metadata
from typing import Any

import httpx
from perplexity import AsyncPerplexity as SDKAsyncPerplexity
from perplexity import Perplexity as SDKPerplexity
from pytest_mock import MockerFixture

from langchain_perplexity import (
    ChatPerplexity,
    PerplexityEmbeddings,
    PerplexitySearchResults,
    PerplexitySearchRetriever,
)

_ATTRIBUTION_HEADER = "X-Pplx-Integration"
_EXPECTED_ATTRIBUTION = f"langchain/{metadata.version('langchain-perplexity')}"


def _encode_int8(values: list[int]) -> str:
    raw = struct.pack(f"<{len(values)}b", *values)
    return base64.b64encode(raw).decode("ascii")


def _make_response(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/chat/completions":
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "delta": {"content": "", "role": "assistant"},
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "ok", "role": "assistant"},
                    }
                ],
                "created": 0,
                "model": "sonar",
            },
            request=request,
        )
    if request.url.path == "/v1/embeddings":
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "embedding": _encode_int8([1, 2, 3]),
                        "index": 0,
                        "object": "embedding",
                    }
                ],
                "model": "pplx-embed-v1-4b",
                "object": "list",
            },
            request=request,
        )
    if request.url.path == "/search":
        return httpx.Response(
            200,
            json={
                "id": "search-test",
                "results": [
                    {
                        "date": "2026-01-01",
                        "last_updated": "2026-01-02",
                        "snippet": "Test snippet",
                        "title": "Test title",
                        "url": "https://example.com",
                    }
                ],
            },
            request=request,
        )

    return httpx.Response(404, request=request)


def _patch_sync_client(
    mocker: MockerFixture,
    target: str,
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _make_response(request)

    def make_client(**kwargs: Any) -> SDKPerplexity:
        return SDKPerplexity(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    mocker.patch(target, side_effect=make_client)
    return requests


def _patch_sync_and_async_clients(
    mocker: MockerFixture,
    *,
    async_target: str,
    sync_target: str,
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _make_response(request)

    def make_client(**kwargs: Any) -> SDKPerplexity:
        return SDKPerplexity(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    def make_async_client(**kwargs: Any) -> SDKAsyncPerplexity:
        return SDKAsyncPerplexity(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    mocker.patch(sync_target, side_effect=make_client)
    mocker.patch(async_target, side_effect=make_async_client)
    return requests


def _assert_attribution_header_sent(requests: list[httpx.Request]) -> None:
    assert requests
    assert requests[0].headers[_ATTRIBUTION_HEADER] == _EXPECTED_ATTRIBUTION


def test_chat_sends_attribution_header(mocker: MockerFixture) -> None:
    requests = _patch_sync_and_async_clients(
        mocker,
        sync_target="langchain_perplexity.chat_models.Perplexity",
        async_target="langchain_perplexity.chat_models.AsyncPerplexity",
    )

    chat = ChatPerplexity(api_key="test", model="sonar")
    chat.invoke("hello")

    _assert_attribution_header_sent(requests)


def test_embeddings_send_attribution_header(mocker: MockerFixture) -> None:
    requests = _patch_sync_and_async_clients(
        mocker,
        sync_target="langchain_perplexity.embeddings.Perplexity",
        async_target="langchain_perplexity.embeddings.AsyncPerplexity",
    )

    embeddings = PerplexityEmbeddings(pplx_api_key="test")
    embeddings.embed_query("hello")

    _assert_attribution_header_sent(requests)


def test_search_retriever_sends_attribution_header(mocker: MockerFixture) -> None:
    requests = _patch_sync_client(mocker, "langchain_perplexity._utils.Perplexity")

    retriever = PerplexitySearchRetriever(pplx_api_key="test")
    retriever.invoke("hello")

    _assert_attribution_header_sent(requests)


def test_search_tool_sends_attribution_header(mocker: MockerFixture) -> None:
    requests = _patch_sync_client(mocker, "langchain_perplexity._utils.Perplexity")

    tool = PerplexitySearchResults(pplx_api_key="test")
    tool.invoke("hello")

    _assert_attribution_header_sent(requests)

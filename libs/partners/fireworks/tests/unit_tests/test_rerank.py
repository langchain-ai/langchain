from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_fireworks import FireworksRerank


class FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        path: str,
        *,
        cast_to: type[dict[str, Any]],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append({"path": path, "cast_to": cast_to, "body": body})
        return self.response


class FakeAsyncClient(FakeClient):
    async def post(  # type: ignore[override]
        self,
        path: str,
        *,
        cast_to: type[dict[str, Any]],
        body: dict[str, Any],
    ) -> dict[str, Any]:
        return super().post(path, cast_to=cast_to, body=body)


def _reranker(
    response: dict[str, Any], **kwargs: Any
) -> tuple[FireworksRerank, FakeClient, FakeAsyncClient]:
    client = FakeClient(response)
    async_client = FakeAsyncClient(response)
    reranker = FireworksRerank(client=client, async_client=async_client, **kwargs)
    return reranker, client, async_client


def test_missing_api_key_without_clients() -> None:
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY is required"):
        FireworksRerank(model="reranker", fireworks_api_key=None)


def test_missing_api_key_with_only_sync_client() -> None:
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY is required"):
        FireworksRerank(
            model="reranker",
            client=FakeClient({"data": []}),
            fireworks_api_key=None,
        )


def test_rerank_posts_fireworks_payload() -> None:
    reranker, client, _ = _reranker(
        {"data": [{"index": 1, "relevance_score": 0.9}]},
        model="fireworks/qwen3-reranker-8b",
    )

    result = reranker.rerank(
        [Document("first"), Document("second")],
        "the query",
        top_n=1,
    )

    assert result == [{"index": 1, "relevance_score": 0.9}]
    assert client.calls == [
        {
            "path": "/rerank",
            "cast_to": dict[str, Any],
            "body": {
                "model": "fireworks/qwen3-reranker-8b",
                "query": "the query",
                "documents": ["first", "second"],
                "return_documents": False,
                "top_n": 1,
            },
        }
    ]


async def test_arerank_posts_fireworks_payload() -> None:
    reranker, client, async_client = _reranker(
        {"data": [{"index": 1, "relevance_score": 0.9}]},
        model="fireworks/qwen3-reranker-8b",
    )

    result = await reranker.arerank(
        [Document("first"), Document("second")],
        "the query",
        top_n=1,
    )

    assert result == [{"index": 1, "relevance_score": 0.9}]
    assert async_client.calls == [
        {
            "path": "/rerank",
            "cast_to": dict[str, Any],
            "body": {
                "model": "fireworks/qwen3-reranker-8b",
                "query": "the query",
                "documents": ["first", "second"],
                "return_documents": False,
                "top_n": 1,
            },
        }
    ]
    assert client.calls == []


def test_rerank_serializes_mappings_and_rank_fields() -> None:
    reranker, client, _ = _reranker({"data": []}, model="reranker")

    reranker.rerank(
        [{"title": "keep", "body": "keep", "ignored": "drop"}],
        "query",
        rank_fields=["title", "body"],
        top_n=None,
    )

    assert client.calls[0]["body"] == {
        "model": "reranker",
        "query": "query",
        "documents": ['{"title": "keep", "body": "keep"}'],
        "return_documents": False,
    }


async def test_arerank_serializes_mappings_and_rank_fields() -> None:
    reranker, _, async_client = _reranker({"data": []}, model="reranker")

    await reranker.arerank(
        [{"title": "keep", "body": "keep", "ignored": "drop"}],
        "query",
        rank_fields=["title", "body"],
        top_n=None,
    )

    assert async_client.calls[0]["body"] == {
        "model": "reranker",
        "query": "query",
        "documents": ['{"title": "keep", "body": "keep"}'],
        "return_documents": False,
    }


def test_rerank_empty_documents_does_not_call_client() -> None:
    reranker, client, _ = _reranker({"data": []}, model="reranker")

    assert reranker.rerank([], "query") == []
    assert client.calls == []


async def test_arerank_empty_documents_does_not_call_client() -> None:
    reranker, _, async_client = _reranker({"data": []}, model="reranker")

    assert await reranker.arerank([], "query") == []
    assert async_client.calls == []


def test_compress_documents_preserves_metadata_and_adds_score() -> None:
    reranker, _, _ = _reranker(
        {
            "data": [
                {"index": 1, "relevance_score": 0.8},
                {"index": 0, "relevance_score": 0.4},
            ]
        },
        model="reranker",
    )
    documents = [
        Document("first", metadata={"nested": {"value": 1}}),
        Document("second", metadata={"source": "test"}),
    ]

    result = reranker.compress_documents(documents, "query")

    assert [document.page_content for document in result] == ["second", "first"]
    assert result[0].metadata == {"source": "test", "relevance_score": 0.8}
    assert result[1].metadata == {
        "nested": {"value": 1},
        "relevance_score": 0.4,
    }


async def test_acompress_documents_preserves_metadata_and_adds_score() -> None:
    reranker, client, async_client = _reranker(
        {
            "data": [
                {"index": 1, "relevance_score": 0.8},
                {"index": 0, "relevance_score": 0.4},
            ]
        },
        model="reranker",
    )
    documents = [
        Document("first", metadata={"nested": {"value": 1}}),
        Document("second", metadata={"source": "test"}),
    ]

    result = await reranker.acompress_documents(documents, "query")

    assert [document.page_content for document in result] == ["second", "first"]
    assert result[0].metadata == {"source": "test", "relevance_score": 0.8}
    assert result[1].metadata == {
        "nested": {"value": 1},
        "relevance_score": 0.4,
    }
    assert documents[1].metadata == {"source": "test"}
    # The async path must not fall back to the sync client via `run_in_executor`.
    assert client.calls == []
    assert len(async_client.calls) == 1

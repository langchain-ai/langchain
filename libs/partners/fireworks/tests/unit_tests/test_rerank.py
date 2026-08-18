from typing import Any

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


def test_rerank_posts_fireworks_payload() -> None:
    client = FakeClient({"data": [{"index": 1, "relevance_score": 0.9}]})
    reranker = FireworksRerank(model="fireworks/qwen3-reranker-8b", client=client)

    result = reranker.rerank(
        [Document("first"), Document("second")],
        "the query",
        top_n=1,
    )

    assert result == [{"index": 1, "relevance_score": 0.9}]
    assert client.calls == [
        {
            "path": "/rerank",
            "cast_to": dict,
            "body": {
                "model": "fireworks/qwen3-reranker-8b",
                "query": "the query",
                "documents": ["first", "second"],
                "return_documents": False,
                "top_n": 1,
            },
        }
    ]


def test_rerank_serializes_mappings_and_rank_fields() -> None:
    client = FakeClient({"data": []})
    reranker = FireworksRerank(model="reranker", client=client)

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


def test_rerank_empty_documents_does_not_call_client() -> None:
    client = FakeClient({"data": []})
    reranker = FireworksRerank(model="reranker", client=client)

    assert reranker.rerank([], "query") == []
    assert client.calls == []


def test_compress_documents_preserves_metadata_and_adds_score() -> None:
    client = FakeClient(
        {
            "data": [
                {"index": 1, "relevance_score": 0.8},
                {"index": 0, "relevance_score": 0.4},
            ]
        }
    )
    reranker = FireworksRerank(model="reranker", client=client)
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

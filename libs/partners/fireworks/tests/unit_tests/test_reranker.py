"""Unit tests for FireworksRerank."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr, ValidationError
from typing_extensions import Self

from langchain_fireworks import FireworksRerank

MODEL_NAME = "fireworks/qwen3-reranker-8b"


def _make_reranker(**kwargs: Any) -> FireworksRerank:
    defaults: dict[str, Any] = {"model": MODEL_NAME, "api_key": "fake-key"}
    defaults.update(kwargs)
    return FireworksRerank(**defaults)  # type: ignore[arg-type]


def _mock_rerank_response(
    indexes: list[int] | None = None,
    scores: list[float] | None = None,
) -> dict[str, Any]:
    indexes = indexes or [0, 1]
    scores = scores or [0.9, 0.5]
    return {
        "object": "list",
        "model": MODEL_NAME,
        "data": [
            {"index": idx, "relevance_score": score}
            for idx, score in zip(indexes, scores, strict=False)
        ],
        "usage": {"prompt_tokens": 42, "total_tokens": 42},
    }


class _MockAiohttpResponse:
    """Mock aiohttp response for async rerank tests."""

    status: int = 200

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    async def json(self) -> dict[str, Any]:
        return self._data

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


class _MockAiohttpSession:
    """Mock aiohttp ClientSession returning a configured response."""

    def __init__(self, response: _MockAiohttpResponse) -> None:
        self._response = response

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def post(self, *args: Any, **kwargs: Any) -> _MockAiohttpResponse:
        return self._response


class TestFireworksRerankInit:
    def test_default_model(self) -> None:
        reranker = _make_reranker()
        assert reranker.model == MODEL_NAME

    def test_api_key_is_secret_str(self) -> None:
        reranker = _make_reranker()
        assert isinstance(reranker.fireworks_api_key, SecretStr)
        assert reranker.fireworks_api_key.get_secret_value() == "fake-key"

    def test_api_key_alias(self) -> None:
        reranker = FireworksRerank(model=MODEL_NAME, api_key="via-alias")  # type: ignore[arg-type]
        assert reranker.fireworks_api_key.get_secret_value() == "via-alias"

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FIREWORKS_API_KEY", "env-api-key")
        reranker = FireworksRerank(model=MODEL_NAME)
        assert reranker.fireworks_api_key.get_secret_value() == "env-api-key"

    def test_default_top_n_is_none(self) -> None:
        reranker = _make_reranker()
        assert reranker.top_n is None

    def test_default_return_documents_is_true(self) -> None:
        reranker = _make_reranker()
        assert reranker.return_documents is True

    def test_default_task_is_none(self) -> None:
        reranker = _make_reranker()
        assert reranker.task is None

    def test_default_timeout_is_30(self) -> None:
        reranker = _make_reranker()
        assert reranker.timeout == 30

    def test_top_n_can_be_set(self) -> None:
        reranker = _make_reranker(top_n=5)
        assert reranker.top_n == 5

    def test_task_can_be_set(self) -> None:
        reranker = _make_reranker(task="classify relevance")
        assert reranker.task == "classify relevance"

    def test_extra_field_is_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            FireworksRerank(model=MODEL_NAME, api_key="key", unknown_field="bad")  # type: ignore[arg-type, call-arg]

    def test_api_key_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
            FireworksRerank(model=MODEL_NAME)

    def test_model_can_be_overridden(self) -> None:
        reranker = _make_reranker(model="fireworks/qwen3-reranker-4b")
        assert reranker.model == "fireworks/qwen3-reranker-4b"


class TestRerank:
    def test_rerank_empty_documents(self) -> None:
        reranker = _make_reranker()
        result = reranker.rerank([], "query")
        assert result == []

    def test_rerank_basic(self) -> None:
        reranker = _make_reranker()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response([0, 1], [0.95, 0.3])
        with patch("requests.post", return_value=mock_response) as mock_post:
            result = reranker.rerank(["doc A", "doc B"], "query")
        assert len(result) == 2
        assert result[0] == {"index": 0, "relevance_score": 0.95}
        assert result[1] == {"index": 1, "relevance_score": 0.3}
        mock_post.assert_called_once()

    def test_rerank_request_payload(self) -> None:
        reranker = _make_reranker()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response()
        with patch("requests.post", return_value=mock_response) as mock_post:
            reranker.rerank(["doc A", "doc B"], "query")
        called_kwargs = mock_post.call_args.kwargs
        assert called_kwargs["json"]["model"] == MODEL_NAME
        assert called_kwargs["json"]["query"] == "query"
        assert called_kwargs["json"]["documents"] == ["doc A", "doc B"]
        assert called_kwargs["json"]["return_documents"] is True
        assert "top_n" not in called_kwargs["json"]

    def test_rerank_request_url(self) -> None:
        """Verify the complete Fireworks rerank API endpoint URL."""
        reranker = _make_reranker()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response()
        with patch("requests.post", return_value=mock_response) as mock_post:
            reranker.rerank(["doc A"], "query")
        assert (
            mock_post.call_args.kwargs["url"]
            == "https://api.fireworks.ai/inference/v1/rerank"
        )

    def test_rerank_request_with_top_n(self) -> None:
        reranker = _make_reranker(top_n=3)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response()
        with patch("requests.post", return_value=mock_response) as mock_post:
            reranker.rerank(["doc A", "doc B", "doc C", "doc D"], "query")
        assert mock_post.call_args.kwargs["json"]["top_n"] == 3

    def test_rerank_request_with_task(self) -> None:
        reranker = _make_reranker(task="classify relevance")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response()
        with patch("requests.post", return_value=mock_response) as mock_post:
            reranker.rerank(["doc A"], "query")
        assert mock_post.call_args.kwargs["json"]["task"] == "classify relevance"

    def test_rerank_with_document_objects(self) -> None:
        reranker = _make_reranker()
        docs = [
            Document("page content 1", metadata={"source": "a"}),
            Document("page content 2", metadata={"source": "b"}),
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response()
        with patch("requests.post", return_value=mock_response) as mock_post:
            reranker.rerank(docs, "query")
        assert mock_post.call_args.kwargs["json"]["documents"] == [
            "page content 1",
            "page content 2",
        ]

    def test_rerank_overrides_model(self) -> None:
        reranker = _make_reranker()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response()
        with patch("requests.post", return_value=mock_response) as mock_post:
            reranker.rerank(["doc A"], "query", model="custom-model")
        assert mock_post.call_args.kwargs["json"]["model"] == "custom-model"

    def test_rerank_overrides_top_n(self) -> None:
        reranker = _make_reranker(top_n=5)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response()
        with patch("requests.post", return_value=mock_response) as mock_post:
            reranker.rerank(["doc A", "doc B", "doc C"], "query", top_n=2)
        assert mock_post.call_args.kwargs["json"]["top_n"] == 2


class TestRerankErrorHandling:
    def test_rerank_500_error(self) -> None:
        reranker = _make_reranker()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal error"
        with (
            patch("requests.post", return_value=mock_response),
            pytest.raises(Exception, match="Fireworks Server: Error 500"),
        ):
            reranker.rerank(["doc A"], "query")

    def test_rerank_400_error(self) -> None:
        reranker = _make_reranker()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        with (
            patch("requests.post", return_value=mock_response),
            pytest.raises(ValueError, match="Fireworks received an invalid payload"),
        ):
            reranker.rerank(["doc A"], "query")

    def test_rerank_unexpected_status(self) -> None:
        reranker = _make_reranker()
        mock_response = MagicMock()
        mock_response.status_code = 301
        mock_response.text = "Redirect"
        with (
            patch("requests.post", return_value=mock_response),
            pytest.raises(Exception, match="unexpected response"),
        ):
            reranker.rerank(["doc A"], "query")


class TestCompressDocuments:
    def test_compress_documents_empty(self) -> None:
        reranker = _make_reranker()
        result = reranker.compress_documents([], "query")
        assert result == []

    def test_compress_documents_basic(self) -> None:
        reranker = _make_reranker()
        docs = [
            Document("content A", metadata={"source": "a"}),
            Document("content B", metadata={"source": "b"}),
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response([1, 0], [0.95, 0.3])
        with patch("requests.post", return_value=mock_response):
            result = reranker.compress_documents(docs, "query")
        assert len(result) == 2
        assert result[0].page_content == "content B"
        assert result[0].metadata["source"] == "b"
        assert result[0].metadata["relevance_score"] == 0.95
        assert result[1].page_content == "content A"
        assert result[1].metadata["source"] == "a"
        assert result[1].metadata["relevance_score"] == 0.3

    def test_compress_documents_preserves_original_metadata(self) -> None:
        reranker = _make_reranker()
        docs = [
            Document("content", metadata={"source": "a", "key": "value"}),
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response([0], [0.9])
        with patch("requests.post", return_value=mock_response):
            result = reranker.compress_documents(docs, "query")
        assert result[0].metadata["source"] == "a"
        assert result[0].metadata["key"] == "value"
        assert result[0].metadata["relevance_score"] == 0.9

    def test_compress_documents_does_not_mutate_input(self) -> None:
        reranker = _make_reranker()
        docs = [
            Document("content A", metadata={"source": "a"}),
        ]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _mock_rerank_response([0], [0.9])
        with patch("requests.post", return_value=mock_response):
            reranker.compress_documents(docs, "query")
        assert "relevance_score" not in docs[0].metadata


class TestACompressDocuments:
    @pytest.mark.asyncio
    @pytest.mark.enable_socket
    async def test_acompress_documents_empty(self) -> None:
        reranker = _make_reranker()
        # Empty documents short-circuit before any HTTP call — no mock needed
        result = await reranker.acompress_documents([], "query")
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.enable_socket
    async def test_acompress_documents_basic(self) -> None:
        reranker = _make_reranker()
        docs = [
            Document("content A", metadata={"source": "a"}),
            Document("content B", metadata={"source": "b"}),
        ]

        response = _MockAiohttpResponse(_mock_rerank_response([1, 0], [0.95, 0.3]))

        with patch(
            "langchain_fireworks.reranker.ClientSession",
            return_value=_MockAiohttpSession(response),
        ):
            result = await reranker.acompress_documents(docs, "query")

        assert len(result) == 2
        assert result[0].page_content == "content B"
        assert result[0].metadata["relevance_score"] == 0.95
        assert result[1].page_content == "content A"
        assert result[1].metadata["relevance_score"] == 0.3

    @pytest.mark.asyncio
    @pytest.mark.enable_socket
    async def test_acompress_documents_preserves_metadata(self) -> None:
        reranker = _make_reranker()
        docs = [
            Document("content A", metadata={"source": "a", "key": "value"}),
        ]

        response = _MockAiohttpResponse(_mock_rerank_response([0], [0.9]))

        with patch(
            "langchain_fireworks.reranker.ClientSession",
            return_value=_MockAiohttpSession(response),
        ):
            result = await reranker.acompress_documents(docs, "query")

        assert result[0].metadata["source"] == "a"
        assert result[0].metadata["key"] == "value"
        assert result[0].metadata["relevance_score"] == 0.9


class TestPayloadBuilding:
    def test_payload_with_none_top_n(self) -> None:
        reranker = _make_reranker(top_n=None)
        payload = reranker._build_payload(["doc A"], "query")
        assert "top_n" not in payload

    def test_payload_with_top_n(self) -> None:
        reranker = _make_reranker(top_n=3)
        payload = reranker._build_payload(["doc A"], "query")
        assert payload["top_n"] == 3

    def test_payload_with_task(self) -> None:
        reranker = _make_reranker(task="classify relevance")
        payload = reranker._build_payload(["doc A"], "query")
        assert payload["task"] == "classify relevance"

    def test_payload_without_task(self) -> None:
        reranker = _make_reranker(task=None)
        payload = reranker._build_payload(["doc A"], "query")
        assert "task" not in payload

    def test_payload_override_model(self) -> None:
        reranker = _make_reranker()
        payload = reranker._build_payload(["doc A"], "query", model="custom-model")
        assert payload["model"] == "custom-model"

    def test_payload_override_top_n(self) -> None:
        reranker = _make_reranker(top_n=5)
        payload = reranker._build_payload(["doc A"], "query", top_n=2)
        assert payload["top_n"] == 2

    def test_payload_override_task(self) -> None:
        reranker = _make_reranker(task="old")
        payload = reranker._build_payload(["doc A"], "query", task="new")
        assert payload["task"] == "new"

"""Unit tests for AgentDBRetriever."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from langchain_agentdb import AgentDBRetriever

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ITEM: dict[str, Any] = {
    "id": "abc-123",
    "title": "AI Safety Breakthrough at DeepMind",
    "content_type": "article",
    "summary": "DeepMind researchers published findings on scalable reward modelling.",
    "body": {
        "key_points": [
            "Reward models can be trained with 10x less data",
            "New evaluation benchmark released",
        ],
        "source_url": "https://deepmind.com/research/example",
    },
    "tags": ["ai-safety", "deepmind", "reinforcement-learning"],
    "confidence": 0.92,
    "relevance_score": 0.88,
    "published_at": "2026-04-18 07:00:00",
}

SAMPLE_RESPONSE: dict[str, Any] = {
    "items": [SAMPLE_ITEM],
    "total": 1,
    "page": 1,
    "has_more": False,
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_requires_api_key_when_env_not_set() -> None:
    """ValueError raised when no key is available."""
    env_backup = os.environ.pop("AGENTDB_API_KEY", None)
    try:
        retriever = AgentDBRetriever()
        with pytest.raises(ValueError, match="No AgentDB API key"):
            retriever._resolved_api_key()
    finally:
        if env_backup is not None:
            os.environ["AGENTDB_API_KEY"] = env_backup


def test_resolves_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTDB_API_KEY", "agentdb-from-env")
    retriever = AgentDBRetriever()
    assert retriever._resolved_api_key() == "agentdb-from-env"


def test_resolves_api_key_from_param(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTDB_API_KEY", raising=False)
    retriever = AgentDBRetriever(api_key="agentdb-direct")
    assert retriever._resolved_api_key() == "agentdb-direct"


def test_default_mode_is_search() -> None:
    retriever = AgentDBRetriever(api_key="k")
    assert retriever.mode == "search"


# ---------------------------------------------------------------------------
# Parameter building
# ---------------------------------------------------------------------------


def test_build_params_search() -> None:
    retriever = AgentDBRetriever(api_key="k", k=5)
    url, params = retriever._build_params("AI news")
    assert "/search" in url
    assert params["q"] == "AI news"
    assert params["limit"] == 5


def test_build_params_latest_defaults() -> None:
    retriever = AgentDBRetriever(api_key="k", mode="latest", k=20)
    url, params = retriever._build_params("")
    assert "/latest" in url
    assert params["limit"] == 20
    assert "content_type" not in params
    assert "tags" not in params


def test_build_params_latest_with_filters() -> None:
    retriever = AgentDBRetriever(
        api_key="k", mode="latest", content_type="podcast", tags="ai,startups"
    )
    _, params = retriever._build_params("")
    assert params["content_type"] == "podcast"
    assert params["tags"] == "ai,startups"


def test_k_capped_at_50_for_search() -> None:
    retriever = AgentDBRetriever(api_key="k", mode="search", k=999)
    _, params = retriever._build_params("test")
    assert params["limit"] == 50


def test_k_capped_at_100_for_latest() -> None:
    retriever = AgentDBRetriever(api_key="k", mode="latest", k=999)
    _, params = retriever._build_params("")
    assert params["limit"] == 100


def test_search_raises_on_empty_query() -> None:
    retriever = AgentDBRetriever(api_key="k", mode="search")
    with pytest.raises(ValueError, match="non-empty query"):
        retriever._build_params("   ")


# ---------------------------------------------------------------------------
# Document conversion
# ---------------------------------------------------------------------------


def test_items_to_documents_basic() -> None:
    retriever = AgentDBRetriever(api_key="k")
    docs = retriever._items_to_documents([SAMPLE_ITEM])
    assert len(docs) == 1
    doc = docs[0]
    assert "DeepMind" in doc.page_content
    assert "Key points:" in doc.page_content
    assert "Reward models" in doc.page_content
    assert doc.metadata["id"] == "abc-123"
    assert doc.metadata["confidence"] == 0.92
    assert "ai-safety" in doc.metadata["tags"]


def test_items_to_documents_min_confidence_excludes() -> None:
    retriever = AgentDBRetriever(api_key="k", min_confidence=0.95)
    docs = retriever._items_to_documents([SAMPLE_ITEM])
    assert len(docs) == 0


def test_items_to_documents_min_confidence_passes() -> None:
    retriever = AgentDBRetriever(api_key="k", min_confidence=0.90)
    docs = retriever._items_to_documents([SAMPLE_ITEM])
    assert len(docs) == 1


def test_items_to_documents_missing_body() -> None:
    item = {**SAMPLE_ITEM, "body": None}
    retriever = AgentDBRetriever(api_key="k")
    docs = retriever._items_to_documents([item])
    assert len(docs) == 1
    assert docs[0].metadata["key_points"] == []


def test_items_to_documents_empty_list() -> None:
    assert AgentDBRetriever(api_key="k")._items_to_documents([]) == []


# ---------------------------------------------------------------------------
# Sync retrieval
# ---------------------------------------------------------------------------


def test_get_relevant_documents_search() -> None:
    retriever = AgentDBRetriever(api_key="agentdb-test", mode="search")
    with patch("httpx.Client") as mock_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_cls.return_value = mock_client

        docs = retriever._get_relevant_documents("AI safety", run_manager=MagicMock())

    assert len(docs) == 1
    assert "/search" in mock_client.get.call_args[0][0]


def test_get_relevant_documents_latest() -> None:
    retriever = AgentDBRetriever(api_key="agentdb-test", mode="latest")
    with patch("httpx.Client") as mock_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_cls.return_value = mock_client

        docs = retriever._get_relevant_documents("", run_manager=MagicMock())

    assert len(docs) == 1
    assert "/latest" in mock_client.get.call_args[0][0]


def test_http_error_propagates() -> None:
    retriever = AgentDBRetriever(api_key="k", mode="latest")
    with patch("httpx.Client") as mock_cls:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock()
        )
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_cls.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            retriever._get_relevant_documents("", run_manager=MagicMock())


# ---------------------------------------------------------------------------
# Async retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_relevant_documents_search() -> None:
    retriever = AgentDBRetriever(api_key="agentdb-test", mode="search")
    with patch("httpx.AsyncClient") as mock_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client

        docs = await retriever._aget_relevant_documents(
            "AI regulation", run_manager=MagicMock()
        )

    assert len(docs) == 1
    assert "/search" in mock_client.get.call_args[0][0]


@pytest.mark.asyncio
async def test_aget_relevant_documents_latest() -> None:
    retriever = AgentDBRetriever(api_key="agentdb-test", mode="latest")
    with patch("httpx.AsyncClient") as mock_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = SAMPLE_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client

        docs = await retriever._aget_relevant_documents("", run_manager=MagicMock())

    assert len(docs) == 1
    assert "/latest" in mock_client.get.call_args[0][0]

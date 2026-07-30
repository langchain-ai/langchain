"""Unit tests for Exa search tools and retrievers."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from exa_py import AsyncExa, Exa
from pydantic import BaseModel, ConfigDict, Field

from langchain_exa import (
    ExaAgent,
    ExaAnswer,
    ExaContents,
    ExaFindSimilarResults,
    ExaSearchResults,
    ExaSearchRetriever,
)


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                text="content",
                title="title",
                url="https://example.com",
                id="id",
                score=1.0,
                published_date=None,
                author=None,
                highlights=None,
                highlight_scores=None,
                summary=None,
            )
        ]
    )


def test_search_tool_defaults_and_freshness() -> None:
    """The search tool defaults to text content and automatic search."""
    client = Exa(api_key="test")
    search = MagicMock(return_value=_response())
    setattr(client, "search", search)
    tool = ExaSearchResults(client=client)

    tool.invoke({"query": "test", "max_age_hours": 24})

    search.assert_called_once()
    kwargs = search.call_args.kwargs
    assert kwargs["contents"]["text"] is True
    assert kwargs["type"] == "auto"
    assert kwargs["contents"]["max_age_hours"] == 24


def test_search_tool_forwards_independent_content_options() -> None:
    """Content options are independent, so each requested option is forwarded."""
    client = Exa(api_key="test")
    search = MagicMock(return_value=_response())
    setattr(client, "search", search)
    tool = ExaSearchResults(client=client)

    tool.invoke(
        {
            "query": "test",
            "text_contents_options": True,
            "highlights": True,
            "summary": True,
        }
    )

    contents = search.call_args.kwargs["contents"]
    assert contents["text"] is True
    assert contents["highlights"] is True
    assert contents["summary"] is True


def test_search_tool_requests_only_highlights() -> None:
    """Asking for highlights alone does not add an unrequested text option."""
    client = Exa(api_key="test")
    search = MagicMock(return_value=_response())
    setattr(client, "search", search)
    tool = ExaSearchResults(client=client)

    tool.invoke({"query": "test", "highlights": True})

    contents = search.call_args.kwargs["contents"]
    assert contents["highlights"] is True
    assert "text" not in contents


def test_search_tool_warns_on_use_autoprompt() -> None:
    """`use_autoprompt` is accepted but deprecated and never sent to Exa."""
    client = Exa(api_key="test")
    search = MagicMock(return_value=_response())
    setattr(client, "search", search)
    tool = ExaSearchResults(client=client)

    with pytest.warns(DeprecationWarning, match="use_autoprompt"):
        tool.invoke({"query": "test", "use_autoprompt": True})

    assert "use_autoprompt" not in search.call_args.kwargs


@pytest.mark.asyncio
async def test_search_tool_async() -> None:
    """The search tool uses the asynchronous Exa client."""
    async_client = AsyncExa(api_key="test")
    search = AsyncMock(return_value=_response())
    setattr(async_client, "search", search)
    tool = ExaSearchResults(async_client=async_client)

    await tool.ainvoke({"query": "test", "max_age_hours": 12})

    search.assert_awaited_once()
    assert search.await_args is not None
    kwargs = search.await_args.kwargs
    assert kwargs["contents"]["max_age_hours"] == 12


def test_retriever_defaults_and_freshness() -> None:
    """The retriever passes automatic search and freshness options."""
    client = Exa(api_key="test")
    search = MagicMock(return_value=_response())
    setattr(client, "search", search)
    retriever = ExaSearchRetriever(client=client, max_age_hours=48)

    assert retriever.invoke("test")[0].page_content == "content"

    kwargs = search.call_args.kwargs
    assert kwargs["contents"]["text"] is True
    assert kwargs["type"] == "auto"
    assert kwargs["contents"]["max_age_hours"] == 48


@pytest.mark.asyncio
async def test_retriever_async() -> None:
    """The retriever uses the asynchronous Exa client."""
    async_client = AsyncExa(api_key="test")
    search = AsyncMock(return_value=_response())
    setattr(async_client, "search", search)
    retriever = ExaSearchRetriever(async_client=async_client)

    docs = await retriever.ainvoke("test")

    assert docs[0].page_content == "content"
    search.assert_awaited_once()


@dataclass
class _AnswerResponse:
    answer: str
    citations: list[object] = field(default_factory=list)
    cost_dollars: object | None = None


@dataclass
class _ContentsResponse:
    results: list[object] = field(default_factory=list)
    statuses: list[object] = field(default_factory=list)


class _AgentRun(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    status: str
    stop_reason: str | None = Field(default=None, alias="stopReason")
    cost_dollars: dict[str, float] | None = Field(default=None, alias="costDollars")
    output: dict[str, object] | None = None


def test_contents_answer_and_agent_tools() -> None:
    """New tools normalize SDK responses to snake_case dicts."""
    client = Exa(api_key="test")
    contents = MagicMock(return_value=_ContentsResponse())
    answer = MagicMock(return_value=_AnswerResponse(answer="yes"))
    run = SimpleNamespace(id="run-1")
    finished = _AgentRun(
        id="run-1",
        status="completed",
        stopReason="completed",
        costDollars={"total": 0.01},
        output={},
    )
    create = MagicMock(return_value=run)
    poll = MagicMock(return_value=finished)
    setattr(client, "get_contents", contents)
    setattr(client, "answer", answer)
    setattr(
        client,
        "agent",
        SimpleNamespace(runs=SimpleNamespace(create=create, poll_until_finished=poll)),
    )

    contents_result = ExaContents(client=client).invoke(
        {"urls": ["https://example.com"]}
    )
    answer_result = ExaAnswer(client=client).invoke({"query": "test"})
    agent_result = ExaAgent(client=client).invoke({"query": "test"})

    assert contents_result["statuses"] == []
    assert answer_result["answer"] == "yes"
    assert answer_result["citations"] == []
    assert agent_result["status"] == "completed"
    assert agent_result["stop_reason"] == "completed"
    assert agent_result["cost_dollars"] == {"total": 0.01}
    assert "stopReason" not in agent_result
    assert "costDollars" not in agent_result
    contents.assert_called_once()
    answer.assert_called_once()
    create.assert_called_once()
    poll.assert_called_once()
    assert contents.call_args.kwargs == {"text": True}


def test_find_similar_emits_deprecation_warning() -> None:
    """Find Similar remains available but warns that callers should prefer Search."""
    client = Exa(api_key="test")
    find_similar = MagicMock(return_value=_response())
    setattr(client, "find_similar_and_contents", find_similar)
    tool = ExaFindSimilarResults(client=client)

    with pytest.warns(DeprecationWarning, match="ExaFindSimilarResults"):
        tool.invoke({"url": "https://example.com", "num_results": 2})

    find_similar.assert_called_once()


def test_contents_forwards_independent_options() -> None:
    """Contents forwards each requested option without dropping any."""
    client = Exa(api_key="test")
    get_contents = MagicMock(return_value={"results": [], "statuses": []})
    setattr(client, "get_contents", get_contents)

    ExaContents(client=client).invoke(
        {"urls": ["https://example.com"], "highlights": True, "summary": True}
    )

    assert get_contents.call_args.kwargs == {"highlights": True, "summary": True}


def test_retriever_requests_only_highlights() -> None:
    """The retriever honours an explicit highlights-only request."""
    client = Exa(api_key="test")
    search = MagicMock(return_value=_response())
    setattr(client, "search", search)
    retriever = ExaSearchRetriever(client=client, highlights=True)

    retriever.invoke("test")

    contents = search.call_args.kwargs["contents"]
    assert contents["highlights"] is True
    assert "text" not in contents

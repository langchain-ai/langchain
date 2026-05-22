"""Unit tests for RePhraseQueryRetriever, including async path.

Regression test for the NotImplementedError raised by
_aget_relevant_documents (issue #37619).
"""
import pytest

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from langchain_classic.retrievers.re_phraser import RePhraseQueryRetriever


class _SyncRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"sync:{query}")]


class _AsyncRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"sync:{query}")]

    async def _aget_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=f"async:{query}")]


def _make_chain(reply: str) -> RunnableLambda:
    return RunnableLambda(lambda q: reply)


def _make_async_chain(reply: str) -> RunnableLambda:
    async def _afn(q: str) -> str:
        return reply

    return RunnableLambda(_afn)


# ---------------------------------------------------------------------------
# Sync path (sanity check)
# ---------------------------------------------------------------------------


def test_sync_get_relevant_documents():
    retriever = RePhraseQueryRetriever(
        retriever=_SyncRetriever(),
        llm_chain=_make_chain("rephrased"),
    )
    docs = retriever.invoke("original query")
    assert len(docs) == 1
    assert docs[0].page_content == "sync:rephrased"


# ---------------------------------------------------------------------------
# Async path (regression for #37619)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aget_relevant_documents_no_longer_raises():
    """_aget_relevant_documents must not raise NotImplementedError."""
    retriever = RePhraseQueryRetriever(
        retriever=_AsyncRetriever(),
        llm_chain=_make_async_chain("async-rephrased"),
    )
    docs = await retriever.ainvoke("original query")
    assert len(docs) == 1
    assert docs[0].page_content == "async:async-rephrased"


@pytest.mark.asyncio
async def test_aget_uses_rephrased_query():
    """The async path passes the re-phrased (not the original) query to the retriever."""
    retriever = RePhraseQueryRetriever(
        retriever=_AsyncRetriever(),
        llm_chain=_make_async_chain("totally different"),
    )
    docs = await retriever.ainvoke("what is langchain?")
    assert docs[0].page_content == "async:totally different"


@pytest.mark.asyncio
async def test_sync_and_async_agree():
    """Sync and async paths must return the same logical result."""
    chain = _make_chain("same query")

    class _BothRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager=None):
            return [Document(page_content=f"doc:{query}")]

        async def _aget_relevant_documents(self, query, *, run_manager=None):
            return [Document(page_content=f"doc:{query}")]

    retriever = RePhraseQueryRetriever(retriever=_BothRetriever(), llm_chain=chain)
    sync_docs = retriever.invoke("anything")
    async_docs = await retriever.ainvoke("anything")
    assert [d.page_content for d in sync_docs] == [d.page_content for d in async_docs]

"""Regression tests for RePhraseQueryRetriever async support.

_aget_relevant_documents previously raised NotImplementedError instead of
delegating to the LLM chain and inner retriever asynchronously.
"""

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from langchain_classic.retrievers.re_phraser import RePhraseQueryRetriever


class _SyncRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=f"sync:{query}")]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=f"async:{query}")]


@pytest.mark.asyncio
async def test_rephrase_aget_returns_docs():
    """ainvoke on RePhraseQueryRetriever must return documents, not raise."""
    chain = RunnableLambda(lambda q: "rephrased")
    retriever = RePhraseQueryRetriever(retriever=_SyncRetriever(), llm_chain=chain)

    docs = await retriever.ainvoke("original question")

    assert len(docs) == 1
    assert docs[0].page_content == "async:rephrased"


@pytest.mark.asyncio
async def test_rephrase_aget_uses_rephrased_query():
    """The async path must pass the LLM chain output to the inner retriever, not the raw query."""
    chain = RunnableLambda(lambda q: "REPHRASED")
    inner = _SyncRetriever()

    retriever = RePhraseQueryRetriever(retriever=inner, llm_chain=chain)
    docs = await retriever.ainvoke("raw query")

    assert docs[0].page_content == "async:REPHRASED", (
        "Inner retriever should receive the rephrased query, not the original"
    )


@pytest.mark.asyncio
async def test_rephrase_aget_async_chain_called():
    """The async path calls llm_chain.ainvoke, not llm_chain.invoke."""
    ainvoke_called = []

    class _TrackingChain:
        def invoke(self, query, config=None):
            return "sync_rephrase"

        async def ainvoke(self, query, config=None):
            ainvoke_called.append(query)
            return "async_rephrase"

    retriever = RePhraseQueryRetriever(
        retriever=_SyncRetriever(), llm_chain=_TrackingChain()
    )
    await retriever.ainvoke("my question")

    assert len(ainvoke_called) == 1, "llm_chain.ainvoke should have been called once"
    assert ainvoke_called[0] == "my question"


def test_rephrase_sync_still_works():
    """Smoke-test that the synchronous path is unaffected."""
    chain = RunnableLambda(lambda q: "sync_rephrased")
    retriever = RePhraseQueryRetriever(retriever=_SyncRetriever(), llm_chain=chain)

    docs = retriever.invoke("question")

    assert len(docs) == 1
    assert docs[0].page_content == "sync:sync_rephrased"

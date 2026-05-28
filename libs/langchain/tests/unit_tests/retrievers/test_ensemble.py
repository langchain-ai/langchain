import asyncio

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing_extensions import override

from langchain_classic.retrievers.ensemble import EnsembleRetriever


class MockRetriever(BaseRetriever):
    docs: list[Document]

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Return the documents."""
        return self.docs


class WeirdRetriever(BaseRetriever):
    """Retriever that returns non-Document items (mimics misbehaved custom retrievers)."""

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list:  # type: ignore[override]
        return [42]

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
    ) -> list:  # type: ignore[override]
        return [42]


def test_invoke() -> None:
    documents1 = [
        Document(page_content="a", metadata={"id": 1}),
        Document(page_content="b", metadata={"id": 2}),
        Document(page_content="c", metadata={"id": 3}),
    ]
    documents2 = [Document(page_content="b")]

    retriever1 = MockRetriever(docs=documents1)
    retriever2 = MockRetriever(docs=documents2)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2],
        weights=[0.5, 0.5],
        id_key=None,
    )
    ranked_documents = ensemble_retriever.invoke("_")

    # The document with page_content "b" in documents2
    # will be merged with the document with page_content "b"
    # in documents1, so the length of ranked_documents should be 3.
    # Additionally, the document with page_content "b" will be ranked 1st.
    assert len(ranked_documents) == 3
    assert ranked_documents[0].page_content == "b"

    documents1 = [
        Document(page_content="a", metadata={"id": 1}),
        Document(page_content="b", metadata={"id": 2}),
        Document(page_content="c", metadata={"id": 3}),
    ]
    documents2 = [Document(page_content="d")]

    retriever1 = MockRetriever(docs=documents1)
    retriever2 = MockRetriever(docs=documents2)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2],
        weights=[0.5, 0.5],
        id_key=None,
    )
    ranked_documents = ensemble_retriever.invoke("_")

    # The document with page_content "d" in documents2 will not be merged
    # with any document in documents1, so the length of ranked_documents
    # should be 4. The document with page_content "a" and the document
    # with page_content "d" will have the same score, but the document
    # with page_content "a" will be ranked 1st because retriever1 has a smaller index.
    assert len(ranked_documents) == 4
    assert ranked_documents[0].page_content == "a"

    documents1 = [
        Document(page_content="a", metadata={"id": 1}),
        Document(page_content="b", metadata={"id": 2}),
        Document(page_content="c", metadata={"id": 3}),
    ]
    documents2 = [Document(page_content="d", metadata={"id": 2})]

    retriever1 = MockRetriever(docs=documents1)
    retriever2 = MockRetriever(docs=documents2)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2],
        weights=[0.5, 0.5],
        id_key="id",
    )
    ranked_documents = ensemble_retriever.invoke("_")

    # Since id_key is specified, the document with id 2 will be merged.
    # Therefore, the length of ranked_documents should be 3.
    # Additionally, the document with page_content "b" will be ranked 1st.
    assert len(ranked_documents) == 3
    assert ranked_documents[0].page_content == "b"


# ---------------------------------------------------------------------------
# Regression tests for https://github.com/langchain-ai/langchain/issues/37736
# arank_fusion must use the same normalization logic as rank_fusion so that
# both paths behave identically when retrievers return non-Document items.
# ---------------------------------------------------------------------------


def test_arank_fusion_raises_same_error_as_rank_fusion_for_non_documents() -> None:
    """Both sync and async paths should raise the same error for non-Document items.

    Before the fix, arank_fusion tried ``Document(page_content=42)`` which
    Pydantic rejected with a ValidationError. The sync path instead passed
    non-Document items through, eventually raising AttributeError inside
    weighted_reciprocal_rank.  The pydantic ValidationError was the wrong
    error — callers expect consistent behaviour across both paths.
    """
    import pytest

    ensemble = EnsembleRetriever(retrievers=[WeirdRetriever()], weights=[1.0])

    sync_exc: type[Exception] | None = None
    async_exc: type[Exception] | None = None

    try:
        ensemble.invoke("test")
    except Exception as e:
        sync_exc = type(e)

    try:
        asyncio.run(ensemble.ainvoke("test"))
    except Exception as e:
        async_exc = type(e)

    # Both paths must raise (or both must not raise), and the exception type must match.
    # Crucially, the async path must NOT raise a pydantic ValidationError when the
    # sync path would raise something else (AttributeError).
    from pydantic import ValidationError

    assert async_exc is not ValidationError, (
        "arank_fusion raised ValidationError but rank_fusion did not; "
        "the async path must use the same normalization as the sync path."
    )
    assert sync_exc == async_exc, (
        f"sync raised {sync_exc}, async raised {async_exc}; both paths must be consistent"
    )

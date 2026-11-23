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


class BareStringRetriever(BaseRetriever):
    """Retriever that returns bare strings instead of Documents."""

    strings: list[str]

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list:
        return list(self.strings)

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
    ) -> list:
        return list(self.strings)


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


def test_rank_fusion_bare_strings() -> None:
    """Bare strings returned by a retriever should be wrapped into Documents."""
    retriever = BareStringRetriever(strings=["foo", "bar"])
    ensemble = EnsembleRetriever(retrievers=[retriever], weights=[1.0])
    results = ensemble.invoke("_")
    assert all(isinstance(doc, Document) for doc in results)
    assert {doc.page_content for doc in results} == {"foo", "bar"}


async def test_arank_fusion_bare_strings() -> None:
    """arank_fusion should wrap bare strings the same way rank_fusion does."""
    retriever = BareStringRetriever(strings=["foo", "bar"])
    ensemble = EnsembleRetriever(retrievers=[retriever], weights=[1.0])
    results = await ensemble.ainvoke("_")
    assert all(isinstance(doc, Document) for doc in results)
    assert {doc.page_content for doc in results} == {"foo", "bar"}


async def test_arank_fusion_matches_rank_fusion() -> None:
    """Sync and async rank fusion should produce identical results."""
    docs = [
        Document(page_content="alpha", metadata={"id": 1}),
        Document(page_content="beta", metadata={"id": 2}),
    ]
    retriever = MockRetriever(docs=docs)
    ensemble = EnsembleRetriever(retrievers=[retriever], weights=[1.0])

    sync_results = ensemble.invoke("_")
    async_results = await ensemble.ainvoke("_")

    assert [d.page_content for d in sync_results] == [
        d.page_content for d in async_results
    ]

from typing import Optional

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing_extensions import override

from langchain.retrievers.ensemble import EnsembleRetriever


class MockRetriever(BaseRetriever):
    docs: list[Document]

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        """Return the documents"""
        return self.docs


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

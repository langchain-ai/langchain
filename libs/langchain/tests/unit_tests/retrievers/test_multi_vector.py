from typing import Any, Callable

from langchain_core.documents import Document
from typing_extensions import override

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain_classic.storage import InMemoryStore
from tests.unit_tests.indexes.test_indexing import InMemoryVectorStore


class InMemoryVectorstoreWithSearch(InMemoryVectorStore):
    @staticmethod
    def _identity_fn(score: float) -> float:
        return score

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._identity_fn

    @override
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        res = self.store.get(query)
        if res is None:
            return []
        return [res]

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        res = self.store.get(query)
        if res is None:
            return []
        return [(res, 0.8)]


def test_multi_vector_retriever_initialization() -> None:
    vectorstore = InMemoryVectorstoreWithSearch()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        doc_id="doc_id",
    )
    documents = [Document(page_content="test document", metadata={"doc_id": "1"})]
    retriever.vectorstore.add_documents(documents, ids=["1"])
    retriever.docstore.mset(list(zip(["1"], documents)))
    results = retriever.invoke("1")
    assert len(results) > 0
    assert results[0].page_content == "test document"


async def test_multi_vector_retriever_initialization_async() -> None:
    vectorstore = InMemoryVectorstoreWithSearch()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        doc_id="doc_id",
    )
    documents = [Document(page_content="test document", metadata={"doc_id": "1"})]
    await retriever.vectorstore.aadd_documents(documents, ids=["1"])
    await retriever.docstore.amset(list(zip(["1"], documents)))
    results = await retriever.ainvoke("1")
    assert len(results) > 0
    assert results[0].page_content == "test document"


def test_multi_vector_retriever_similarity_search_with_score() -> None:
    documents = [Document(page_content="test document", metadata={"doc_id": "1"})]
    vectorstore = InMemoryVectorstoreWithSearch()
    vectorstore.add_documents(documents, ids=["1"])

    # test with score_threshold = 0.5
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        doc_id="doc_id",
        search_kwargs={"score_threshold": 0.5},
        search_type=SearchType.similarity_score_threshold,
    )
    retriever.docstore.mset(list(zip(["1"], documents)))
    results = retriever.invoke("1")
    assert len(results) == 1
    assert results[0].page_content == "test document"

    # test with score_threshold = 0.9
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        doc_id="doc_id",
        search_kwargs={"score_threshold": 0.9},
        search_type=SearchType.similarity_score_threshold,
    )
    retriever.docstore.mset(list(zip(["1"], documents)))
    results = retriever.invoke("1")
    assert len(results) == 0


async def test_multi_vector_retriever_similarity_search_with_score_async() -> None:
    documents = [Document(page_content="test document", metadata={"doc_id": "1"})]
    vectorstore = InMemoryVectorstoreWithSearch()
    await vectorstore.aadd_documents(documents, ids=["1"])

    # test with score_threshold = 0.5
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        doc_id="doc_id",
        search_kwargs={"score_threshold": 0.5},
        search_type=SearchType.similarity_score_threshold,
    )
    await retriever.docstore.amset(list(zip(["1"], documents)))
    results = retriever.invoke("1")
    assert len(results) == 1
    assert results[0].page_content == "test document"

    # test with score_threshold = 0.9
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        doc_id="doc_id",
        search_kwargs={"score_threshold": 0.9},
        search_type=SearchType.similarity_score_threshold,
    )
    await retriever.docstore.amset(list(zip(["1"], documents)))
    results = retriever.invoke("1")
    assert len(results) == 0

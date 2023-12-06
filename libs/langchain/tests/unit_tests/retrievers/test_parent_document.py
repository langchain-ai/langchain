from typing import Any, List, Sequence

from langchain_core.documents import Document

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tests.unit_tests.indexes.test_indexing import InMemoryVectorStore


class InMemoryVectorstoreWithSearch(InMemoryVectorStore):
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        res = self.store.get(query)
        if res is None:
            return []
        return [res]

    def add_documents(self, documents: Sequence) -> None:
        return super().add_documents(documents, ids=["1"])


def test_parent_document_retriever() -> None:
    vectorstore = InMemoryVectorstoreWithSearch()
    store = InMemoryStore()
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    documents = [Document(page_content="test document", metadata={"doc_id": "1"})]
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    retriever.add_documents(documents, ids=["1"])
    results = retriever.invoke("1")
    assert len(results) > 0
    assert results[0].page_content == "test document"

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters.character import CharacterTextSplitter
from typing_extensions import override

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from tests.unit_tests.indexes.test_indexing import InMemoryVectorStore


class InMemoryVectorstoreWithSearch(InMemoryVectorStore):
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
    def add_documents(self, documents: Sequence[Document], **kwargs: Any) -> list[str]:
        print(documents)  # noqa: T201
        return super().add_documents(
            documents,
            ids=[f"{i}" for i in range(len(documents))],
        )


def test_parent_document_retriever_initialization() -> None:
    vectorstore = InMemoryVectorstoreWithSearch()
    store = InMemoryStore()
    child_splitter = CharacterTextSplitter(chunk_size=400)
    documents = [Document(page_content="test document")]
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )
    retriever.add_documents(documents)
    results = retriever.invoke("0")
    assert len(results) > 0
    assert results[0].page_content == "test document"

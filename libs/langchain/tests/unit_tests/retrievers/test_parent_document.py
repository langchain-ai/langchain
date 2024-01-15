import unittest
from typing import Any, List, Sequence

from langchain_core.documents import Document

from langchain.indexes import SQLRecordManager
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import CharacterTextSplitter
from tests.unit_tests.indexes.test_indexing import InMemoryVectorStore


class InMemoryVectorstoreWithSearch(InMemoryVectorStore):
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        res = self.store.get(query)
        if res is None:
            return []
        return [res]

    def add_documents(self, documents: Sequence[Document], **kwargs: Any) -> List[str]:
        print(documents)
        return super().add_documents(
            documents, ids=[f"{i}" for i in range(len(documents))]
        )


class ParentDocumentRetrieverTests(unittest.TestCase):
    def test_initialization(self) -> None:
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

    def test_initialization_child_splitter_list(self) -> None:
        vectorstore = InMemoryVectorstoreWithSearch()
        store = InMemoryStore()
        child_splitter = CharacterTextSplitter(chunk_size=400)
        documents = [Document(page_content="test document")]
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
        )
        retriever.add_documents(documents, child_splitters=[child_splitter])
        results = retriever.invoke("0")
        assert len(results) > 0
        assert results[0].page_content == "test document"

    def test_initialization_no_child_splitter(self) -> None:
        vectorstore = InMemoryVectorstoreWithSearch()
        store = InMemoryStore()
        documents = [Document(page_content="test document")]
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
        )

        with self.assertRaises(ValueError):
            retriever.add_documents(documents)

    def test_indexing_not_used(self) -> None:
        vectorstore = InMemoryVectorstoreWithSearch()
        store = InMemoryStore()
        child_splitter = CharacterTextSplitter(chunk_size=400)
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )

        add_func = retriever._add_documents  # pylint: disable=W0212
        assert add_func.__name__ != "_index"

    def test_indexing_is_used(self) -> None:
        vectorstore = InMemoryVectorstoreWithSearch()
        store = InMemoryStore()
        child_splitter = CharacterTextSplitter(chunk_size=400)
        record_manager = SQLRecordManager("test", db_url="sqlite:///:memory:")
        record_manager.create_schema()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            index_args=dict(record_manager=record_manager, cleanup=None),
        )

        add_func = retriever._add_documents  # pylint: disable=W0212
        assert add_func.__name__ == "_index"

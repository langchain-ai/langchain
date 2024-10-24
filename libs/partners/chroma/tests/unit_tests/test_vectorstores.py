import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from langchain_chroma.vectorstores import Chroma


# Write our own test for the Chroma vectorstore
class TestChromaSearch:
    @pytest.fixture
    def test_from_documents(self) -> None:
        """Test end to end construction and search."""
        documents = [
            Document(page_content="Dogs are tough.", metadata={"a": 1}),
            Document(page_content="Cats have fluff.", metadata={"b": 1}),
            Document(page_content="What is a sandwich?", metadata={"c": 1}),
            Document(page_content="That fence is purple.", metadata={"d": 1, "e": 2}),
        ]
        vectorstore = Chroma.from_documents(
            collection_name="test_collection",
            documents=documents,
            embedding=FakeEmbeddings(size=10),
        )

        results = vectorstore.similarity_search("What is a sandwich?")
        id = results[0].id
        assert id is not None
        assert vectorstore.get(id)["documents"][0] == results[0].page_content
        newDoc = Document(page_content="This is a new document.", metadata={"x": 1})
        vectorstore.update_document(id, newDoc)
        assert vectorstore.get(id)["documents"][0] == newDoc.page_content

    def test_from_texts(self) -> None:
        """Test end to end construction and search."""
        texts = [
            "Dogs are tough.",
            "Cats have fluff.",
            "What is a sandwich?",
            "That fence is purple.",
        ]
        vectorstore = Chroma.from_texts(
            collection_name="test_collection",
            texts=texts,
            embedding=FakeEmbeddings(size=10),
        )

        results = vectorstore.similarity_search("sandwich")
        assert results is not None, "The similarity_search method returned None"
        assert len(results) > 0, "The similarity_search method returned an empty list"
        assert results[0] is not None, "The first result is None"
        assert results[0].id is not None, "The id of the first result is None"

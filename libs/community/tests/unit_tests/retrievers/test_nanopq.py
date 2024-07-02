import pytest
from langchain_core.documents import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import NanoPQRetriever


class TestNanoPQRetriever:
    @pytest.mark.requires("nanopq")
    def test_from_texts(self) -> None:
        input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
        pq_retriever = NanoPQRetriever.from_texts(
            texts=input_texts, embeddings=FakeEmbeddings(size=100)
        )
        assert len(pq_retriever.texts) == 3

    @pytest.mark.requires("nanopq")
    def test_from_documents(self) -> None:
        input_docs = [
            Document(page_content="I have a pen.", metadata={"page": 1}),
            Document(page_content="Do you have a pen?", metadata={"page": 2}),
            Document(page_content="I have a bag.", metadata={"page": 3}),
        ]
        pq_retriever = NanoPQRetriever.from_documents(
            documents=input_docs, embeddings=FakeEmbeddings(size=100)
        )
        assert pq_retriever.texts == [
            "I have a pen.",
            "Do you have a pen?",
            "I have a bag.",
        ]
        assert pq_retriever.metadatas == [{"page": 1}, {"page": 2}, {"page": 3}]

    @pytest.mark.requires("nanopq")
    def invalid_subspace_error(self) -> None:
        input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
        pq_retriever = NanoPQRetriever.from_texts(
            texts=input_texts, embeddings=FakeEmbeddings(size=43)
        )
        with pytest.raises(RuntimeError):
            pq_retriever.invoke("I have")

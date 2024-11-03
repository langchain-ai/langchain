import pytest
from langchain_core.documents import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers.svm import SVMRetriever


class TestSVMRetriever:
    @pytest.mark.requires("sklearn")
    def test_from_texts(self) -> None:
        input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
        svm_retriever = SVMRetriever.from_texts(
            texts=input_texts, embeddings=FakeEmbeddings(size=100)
        )
        assert len(svm_retriever.texts) == 3

    @pytest.mark.requires("sklearn")
    def test_from_documents(self) -> None:
        input_docs = [
            Document(page_content="I have a pen.", metadata={"foo": "bar"}),
            Document(page_content="Do you have a pen?"),
            Document(page_content="I have a bag."),
        ]
        svm_retriever = SVMRetriever.from_documents(
            documents=input_docs, embeddings=FakeEmbeddings(size=100)
        )
        assert len(svm_retriever.texts) == 3

    @pytest.mark.requires("sklearn")
    def test_metadata_persists(self) -> None:
        input_docs = [
            Document(page_content="I have a pen.", metadata={"foo": "bar"}),
            Document(page_content="How about you?", metadata={"foo": "baz"}),
            Document(page_content="I have a bag.", metadata={"foo": "qux"}),
        ]
        svm_retriever = SVMRetriever.from_documents(
            documents=input_docs, embeddings=FakeEmbeddings(size=100)
        )
        query = "Have anything?"
        output_docs = svm_retriever.invoke(query)
        for doc in output_docs:
            assert "foo" in doc.metadata

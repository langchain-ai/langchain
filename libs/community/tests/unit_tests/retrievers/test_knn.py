from langchain_core.documents import Document

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers.knn import KNNRetriever


class TestKNNRetriever:
    def test_from_texts(self) -> None:
        input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
        knn_retriever = KNNRetriever.from_texts(
            texts=input_texts, embeddings=FakeEmbeddings(size=100)
        )
        assert len(knn_retriever.texts) == 3

    def test_from_documents(self) -> None:
        input_docs = [
            Document(page_content="I have a pen.", metadata={"page": 1}),
            Document(page_content="Do you have a pen?", metadata={"page": 2}),
            Document(page_content="I have a bag.", metadata={"page": 3}),
        ]
        knn_retriever = KNNRetriever.from_documents(
            documents=input_docs, embeddings=FakeEmbeddings(size=100)
        )
        assert knn_retriever.texts == [
            "I have a pen.",
            "Do you have a pen?",
            "I have a bag.",
        ]
        assert knn_retriever.metadatas == [{"page": 1}, {"page": 2}, {"page": 3}]

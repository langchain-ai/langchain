from langchain.embeddings import FakeEmbeddings
from langchain.retrievers.knn import KNNRetriever


class TestKNNRetriever:
    def test_from_texts(self) -> None:
        input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
        knn_retriever = KNNRetriever.from_texts(
            texts=input_texts, embeddings=FakeEmbeddings(size=100)
        )
        assert len(knn_retriever.texts) == 3

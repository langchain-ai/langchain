from typing import List

import numpy as np
import pytest

from langchain.embeddings.base import Embeddings
from langchain.retrievers.knn import KNNRetriever


class TestKNNRetriever:
    class MockEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [np.random.random(size=100).tolist() for _ in texts]

        def embed_query(self, text: str) -> List[float]:
            return np.random.random(size=100).tolist()

        async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
            NotImplemented

        async def aembed_query(self, text: str) -> List[float]:
            NotImplemented

    def test_from_texts(self) -> None:
        input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
        knn_retriever = KNNRetriever.from_texts(
            texts=input_texts, embeddings=self.MockEmbeddings()
        )
        assert len(knn_retriever.texts) == 3

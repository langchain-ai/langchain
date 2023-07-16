from typing import List

import pytest

from langchain.embeddings.base import Embeddings
from langchain.retrievers.svm import SVMRetriever
from langchain.schema import Document


class TestSVMRetriever:
    class MockEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return []

        def embed_query(self, text: str) -> List[float]:
            return []

        async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
            return []

        async def aembed_query(self, text: str) -> List[float]:
            return []

    @pytest.mark.requires("sklearn")
    def test_from_texts(self) -> None:
        input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
        svm_retriever = SVMRetriever.from_texts(
            texts=input_texts, embeddings=self.MockEmbeddings()
        )
        assert len(svm_retriever.texts) == 3

    @pytest.mark.requires("sklearn")
    def test_from_documents(self) -> None:
        input_docs = [
            Document(page_content="I have a pen."),
            Document(page_content="Do you have a pen?"),
            Document(page_content="I have a bag."),
        ]
        svm_retriever = SVMRetriever.from_documents(
            documents=input_docs, embeddings=self.MockEmbeddings()
        )
        assert len(svm_retriever.texts) == 3

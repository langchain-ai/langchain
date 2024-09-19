from typing import List

import requests  # type: ignore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_qdrant import SparseEmbeddings, SparseVector


def qdrant_running_locally() -> bool:
    """Check if Qdrant is running at http://localhost:6333."""

    try:
        response = requests.get("http://localhost:6333", timeout=10.0)
        response_json = response.json()
        return response_json.get("title") == "qdrant - vector search engine"
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def assert_documents_equals(actual: List[Document], expected: List[Document]):  # type: ignore[no-untyped-def]
    assert len(actual) == len(expected)

    for actual_doc, expected_doc in zip(actual, expected):
        assert actual_doc.page_content == expected_doc.page_content

        assert "_id" in actual_doc.metadata
        assert "_collection_name" in actual_doc.metadata

        actual_doc.metadata.pop("_id")
        actual_doc.metadata.pop("_collection_name")

        assert actual_doc.metadata == expected_doc.metadata


class ConsistentFakeEmbeddings(Embeddings):
    """Fake embeddings which remember all the texts seen so far to return consistent
    vectors for the same texts."""

    def __init__(self, dimensionality: int = 10) -> None:
        self.known_texts: List[str] = []
        self.dimensionality = dimensionality

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            vector = [float(1.0)] * (self.dimensionality - 1) + [
                float(self.known_texts.index(text))
            ]
            out_vectors.append(vector)
        return out_vectors

    def embed_query(self, text: str) -> List[float]:
        """Return consistent embeddings for the text, if seen before, or a constant
        one if the text is unknown."""
        return self.embed_documents([text])[0]


class ConsistentFakeSparseEmbeddings(SparseEmbeddings):
    """Fake sparse embeddings which remembers all the texts seen so far "
    "to return consistent vectors for the same texts."""

    def __init__(self, dimensionality: int = 25) -> None:
        self.known_texts: List[str] = []
        self.dimensionality = 25

    def embed_documents(self, texts: List[str]) -> List[SparseVector]:
        """Return consistent embeddings for each text seen so far."""
        out_vectors = []
        for text in texts:
            if text not in self.known_texts:
                self.known_texts.append(text)
            index = self.known_texts.index(text)
            indices = [i + index for i in range(self.dimensionality)]
            values = [1.0] * (self.dimensionality - 1) + [float(index)]
            out_vectors.append(SparseVector(indices=indices, values=values))
        return out_vectors

    def embed_query(self, text: str) -> SparseVector:
        """Return consistent embeddings for the text, "
        "if seen before, or a constant one if the text is unknown."""
        return self.embed_documents([text])[0]

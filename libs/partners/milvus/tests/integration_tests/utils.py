from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

fake_texts = ["foo", "bar", "baz"]


class FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def assert_docs_equal_without_pk(
    docs1: List[Document], docs2: List[Document], pk_field: str = "pk"
) -> None:
    """Assert two lists of Documents are equal, ignoring the primary key field."""
    assert len(docs1) == len(docs2)
    for doc1, doc2 in zip(docs1, docs2):
        assert doc1.page_content == doc2.page_content
        doc1.metadata.pop(pk_field, None)
        doc2.metadata.pop(pk_field, None)
        assert doc1.metadata == doc2.metadata

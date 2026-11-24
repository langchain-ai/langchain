from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
    SparseEmbeddings,
    SparseVector,
)


class _DenseEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _SparseEmbeddings(SparseEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[0], values=[1.0])


def test_hybrid_similarity_search_with_offset() -> None:
    """Hybrid search prefetch limits must include offset for pagination."""
    client = QdrantClient(location=":memory:")
    try:
        client.create_collection(
            "test-hybrid-offset",
            vectors_config={
                "dense": models.VectorParams(size=2, distance=models.Distance.DOT)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        client.upsert(
            "test-hybrid-offset",
            points=[
                models.PointStruct(
                    id=i,
                    vector={
                        "dense": [float(13 - i), 0.0],
                        "sparse": models.SparseVector(
                            indices=[0], values=[float(13 - i)]
                        ),
                    },
                    payload={"page_content": f"doc-{i:02d}", "metadata": {"idx": i}},
                )
                for i in range(1, 13)
            ],
            wait=True,
        )
        store = QdrantVectorStore(
            client=client,
            collection_name="test-hybrid-offset",
            embedding=_DenseEmbeddings(),
            sparse_embedding=_SparseEmbeddings(),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
            distance=models.Distance.DOT,
        )

        page1 = store.similarity_search_with_score("query", k=3, offset=0)
        page2 = store.similarity_search_with_score("query", k=3, offset=3)
        page3 = store.similarity_search_with_score("query", k=3, offset=4)

        assert [doc.metadata["_id"] for doc, _ in page1] == [1, 2, 3]
        assert [doc.metadata["_id"] for doc, _ in page2] == [4, 5, 6]
        assert [doc.metadata["_id"] for doc, _ in page3] == [5, 6, 7]
    finally:
        client.close()

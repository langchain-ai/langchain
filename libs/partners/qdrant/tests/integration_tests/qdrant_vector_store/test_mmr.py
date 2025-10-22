import pytest  # type: ignore[import-not-found]
from langchain_core.documents import Document
from qdrant_client import models

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.qdrant import QdrantVectorStoreError
from tests.integration_tests.common import (
    ConsistentFakeEmbeddings,
    ConsistentFakeSparseEmbeddings,
    assert_documents_equals,
)
from tests.integration_tests.fixtures import qdrant_locations, retrieval_modes


# MMR is supported when dense embeddings are available
# i.e. In Dense and Hybrid retrieval modes
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize(
    "content_payload_key", [QdrantVectorStore.CONTENT_KEY, "test_content"]
)
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "test_metadata"]
)
@pytest.mark.parametrize("retrieval_mode", retrieval_modes(sparse=False))
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
def test_qdrant_mmr_search(
    location: str,
    content_payload_key: str,
    metadata_payload_key: str,
    retrieval_mode: RetrievalMode,
    vector_name: str,
) -> None:
    """Test end to end construction and MRR search."""
    filter_ = models.Filter(
        must=[
            models.FieldCondition(
                key=f"{metadata_payload_key}.page",
                match=models.MatchValue(
                    value=2,
                ),
            ),
        ],
    )

    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        location=location,
        retrieval_mode=retrieval_mode,
        vector_name=vector_name,
        distance=models.Distance.EUCLID,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )
    output = docsearch.max_marginal_relevance_search(
        "foo", k=2, fetch_k=3, lambda_mult=0.0
    )
    assert_documents_equals(
        output,
        [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="baz", metadata={"page": 2}),
        ],
    )

    output = docsearch.max_marginal_relevance_search(
        "foo", k=2, fetch_k=3, lambda_mult=0.0, filter=filter_
    )
    assert_documents_equals(
        output,
        [Document(page_content="baz", metadata={"page": 2})],
    )


# MMR shouldn't work with only sparse retrieval mode
@pytest.mark.parametrize("location", qdrant_locations())
@pytest.mark.parametrize(
    "content_payload_key", [QdrantVectorStore.CONTENT_KEY, "test_content"]
)
@pytest.mark.parametrize(
    "metadata_payload_key", [QdrantVectorStore.METADATA_KEY, "test_metadata"]
)
@pytest.mark.parametrize("retrieval_mode", retrieval_modes(dense=False, hybrid=False))
@pytest.mark.parametrize("vector_name", ["", "my-vector"])
def test_invalid_qdrant_mmr_with_sparse(
    location: str,
    content_payload_key: str,
    metadata_payload_key: str,
    retrieval_mode: RetrievalMode,
    vector_name: str,
) -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = QdrantVectorStore.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
        location=location,
        retrieval_mode=retrieval_mode,
        vector_name=vector_name,
        distance=models.Distance.EUCLID,
        sparse_embedding=ConsistentFakeSparseEmbeddings(),
    )

    with pytest.raises(QdrantVectorStoreError) as excinfo:
        docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3, lambda_mult=0.0)

        expected_message = "does not contain dense vector named"
        assert expected_message in str(excinfo.value)

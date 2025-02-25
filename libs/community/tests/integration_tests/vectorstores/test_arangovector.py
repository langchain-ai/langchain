"""Test ArangoVector functionality."""

from typing import Any, List

from langchain_community.graphs.arangodb_graph import get_arangodb_client
from langchain_community.vectorstores.arangodb_vector import ArangoVector
from langchain_community.vectorstores.utils import DistanceStrategy
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

OS_TOKEN_COUNT = 1536

texts = ["foo", "bar", "baz", "It is the end of the world. Take shelter!", "1.0"]

"""
cd tests/integration_tests/vectorstores/docker-compose
docker-compose -f arangodb.yml up
"""

collection_name = "documents"


def drop_collection(db: Any) -> None:
    """Cleanup document collection"""
    db.delete_collection(collection_name, ignore_missing=True)


class FakeEmbeddingsWithOsDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(i + 1)]
            for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (OS_TOKEN_COUNT - 1) + [float(texts.index(text) + 1)]


def test_arangovector() -> None:
    """Test end to end construction and search."""
    db = get_arangodb_client()
    drop_collection(db)

    docsearch: ArangoVector = ArangoVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        database=db,
        num_centroids=len(texts),
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output[0].page_content == "foo"


def test_arangovector_euclidean() -> None:
    """Test euclidean distance"""
    db = get_arangodb_client()
    drop_collection(db)

    docsearch = ArangoVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        database=db,
        num_centroids=len(texts),
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output[0].page_content == "foo"


def test_arangovector_with_metadatas() -> None:
    """Test end to end construction and search."""
    db = get_arangodb_client()
    drop_collection(db)

    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = ArangoVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        database=db,
        num_centroids=len(texts),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1, return_fields={"page"})
    assert output[0].metadata == {"page": "0"}
    assert output[0].page_content == "foo"


def test_arangovector_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    db = get_arangodb_client()
    drop_collection(db)

    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = ArangoVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        database=db,
        num_centroids=len(texts),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search_with_score(
        "foo", k=1, return_fields={"page"}, use_approx=False
    )
    doc, score = output[0][0], round(output[0][1], 1)

    assert doc.page_content == "foo"
    assert doc.metadata == {"page": "0"}
    assert score == 1.0


def test_arangovector_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    db = get_arangodb_client()
    drop_collection(db)

    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = ArangoVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        database=db,
        num_centroids=len(texts),
        metadatas=metadatas,
    )

    output = docsearch.similarity_search_with_relevance_scores(
        "foo", k=3, use_approx=False, return_fields={"page"}
    )
    assert len(output) == 3
    assert output[0][0].page_content == "foo"
    assert output[0][0].metadata == {"page": "0"}
    assert output[0][1] == 1.0

    assert output[1][0].page_content == "bar"
    assert output[1][0].metadata == {"page": "1"}
    assert output[1][1] > 0.99

    assert output[2][0].page_content == "baz"
    assert output[2][0].metadata == {"page": "2"}
    assert output[2][1] > 0.99


def test_arangovector_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    db = get_arangodb_client()
    drop_collection(db)

    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = ArangoVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        database=db,
        num_centroids=len(texts),
        metadatas=metadatas,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.9999,
            "return_fields": {"page"},
            "use_approx": False,
        },
    )
    output = retriever.invoke("foo")
    assert output[0].page_content == "foo"
    assert output[0].metadata == {"page": "0"}


def test_arangovector_max_marginal_relevance_search() -> None:
    """
    Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==3 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order).
    """
    db = get_arangodb_client()
    drop_collection(db)

    texts = ["-0.124", "+0.127", "+0.25", "+1.0"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = ArangoVector.from_texts(
        texts=texts,
        embedding=FakeEmbeddingsWithOsDimension(),
        database=db,
        num_centroids=len(texts),
        metadatas=metadatas,
    )

    expected_set = {
        ("+1.0", 3),
        ("+0.25", 2),
    }

    output = docsearch.max_marginal_relevance_search(
        "1.0", k=2, fetch_k=3, return_fields={"page"}, use_approx=False
    )
    output_set = {
        (mmr_doc.page_content, mmr_doc.metadata["page"]) for mmr_doc in output
    }
    assert output_set == expected_set

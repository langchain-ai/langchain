"""Test Pinecone functionality."""
import pinecone

from langchain.docstore.document import Document
from langchain.vectorstores.pinecone import Pinecone
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENV")

index = pinecone.Index("langchain-demo")


def test_pinecone() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Pinecone.from_texts(
        texts, FakeEmbeddings(), index_name="langchain-demo", namespace="test"
    )
    output = docsearch.similarity_search("foo", k=1, namespace="test")
    assert output == [Document(page_content="foo")]


def test_pinecone_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Pinecone.from_texts(
        texts,
        FakeEmbeddings(),
        index_name="langchain-demo",
        metadatas=metadatas,
        namespace="test-metadata",
    )
    output = docsearch.similarity_search("foo", k=1, namespace="test-metadata")
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_pinecone_with_scores() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Pinecone.from_texts(
        texts,
        FakeEmbeddings(),
        index_name="langchain-demo",
        metadatas=metadatas,
        namespace="test-metadata-score",
    )
    output = docsearch.similarity_search_with_score(
        "foo", k=3, namespace="test-metadata-score"
    )
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]
    assert scores[0] > scores[1] > scores[2]

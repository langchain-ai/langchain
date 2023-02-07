"""Test Qdrant functionality."""
from langchain.docstore.document import Document
from langchain.vectorstores import Qdrant
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings


def test_qdrant() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Qdrant.from_texts(texts, FakeEmbeddings(), host="localhost")
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_qdrant_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Qdrant.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        host="localhost",
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_qdrant_max_marginal_relevance_search() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = Qdrant.from_texts(
        texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        host="localhost",
    )
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
    ]

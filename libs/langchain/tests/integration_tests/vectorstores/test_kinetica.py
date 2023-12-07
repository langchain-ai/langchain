import os
from typing import List

import pytest
from gpudb import GPUdb

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.kinetica import DistanceStrategy, Kinetica
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

DIMENSIONS = 3
HOST = os.getenv("KINETICA_HOST", "http://127.0.0.1:9191")
USERNAME = os.getenv("KINETICA_USERNAME", "")
PASSWORD = os.getenv("KINETICA_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [[float(1.0)] * (DIMENSIONS - 1) + [float(i)] for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (DIMENSIONS - 1) + [float(0.0)]


@pytest.fixture
def get_db() -> GPUdb:
    options = GPUdb.Options()
    options.username = USERNAME
    options.password = PASSWORD
    options.skip_ssl_cert_verification = True
    return GPUdb(host=HOST, options=options)


# @pytest.mark.skip(reason="Isolating ...")
def test_kinetica(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    texts = ["foo", "bar", "baz"]
    metadatas = [{"text": text} for text in texts]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        dimensions=3,
        collection_name="test_kinetica",
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output[0].page_content == "foo"


# @pytest.mark.skip(reason="Isolating ...")
def test_kinetica_embeddings(get_db: GPUdb) -> None:
    """Test end to end construction with embeddings and search."""
    db = get_db
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Kinetica.from_embeddings(
        db=db,
        text_embeddings=text_embedding_pairs,
        embedding=FakeEmbeddingsWithAdaDimension(),
        dimensions=3,
        collection_name="test_kinetica_embeddings",
        pre_delete_collection=False,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


# @pytest.mark.skip(reason="Isolating ...")
def test_kinetica_with_metadatas(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        dimensions=3,
        collection_name="test_kinetica_with_metadatas",
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_kinetica_with_metadatas_with_scores(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        dimensions=3,
        collection_name="test_kinetica_with_metadatas_with_scores",
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_kinetica_with_filter_match(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        dimensions=3,
        collection_name="test_kinetica_with_filter_match",
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "0"})
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_kinetica_with_filter_distant_match(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        dimensions=3,
        collection_name="test_kinetica_with_filter_distant_match",
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "2"})
    assert output == [(Document(page_content="baz", metadata={"page": "2"}), 2.0)]


@pytest.mark.skip(reason="Filter condition has IN clause")
def test_kinetica_with_filter_in_set(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        dimensions=3,
        collection_name="test_kinetica_with_filter_in_set",
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_score(
        "foo", k=2, filter={"page": {"IN": ["0", "2"]}}
    )
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 0.0),
        (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406),
    ]


def test_kinetica_relevance_score(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        metadatas=metadatas,
        embedding=FakeEmbeddingsWithAdaDimension(),
        dimensions=3,
        collection_name="test_kinetica_relevance_score",
        pre_delete_collection=False,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.29289321881345254),
        (Document(page_content="baz", metadata={"page": "2"}), -0.4142135623730949),
    ]


@pytest.mark.requires("openai")
def test_kinetica_max_marginal_relevance_search(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    openai = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    texts = ["foo", "bar", "baz"]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        embedding=openai,
        dimensions=1536,
        distance_strategy=DistanceStrategy.COSINE,
        collection_name="test_kinetica_max_marginal_relevance_search",
        pre_delete_collection=False,
    )

    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)
    assert output == [Document(page_content="foo")]


def test_kinetica_max_marginal_relevance_search_with_score(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    texts = ["foo", "bar", "baz"]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        embedding=FakeEmbeddingsWithAdaDimension(),
        dimensions=3,
        distance_strategy=DistanceStrategy.EUCLIDEAN,
        collection_name="test_kinetica_max_marginal_relevance_search_with_score",
        pre_delete_collection=False,
    )

    output = docsearch.max_marginal_relevance_search_with_score("foo", k=1, fetch_k=3)
    assert output == [(Document(page_content="foo"), 0.0)]


@pytest.mark.requires("openai")
def test_kinetica_with_openai_embeddings(get_db: GPUdb) -> None:
    """Test end to end construction and search."""
    db = get_db
    if OPENAI_API_KEY == "":
        assert False

    openai = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    texts = ["foo", "bar", "baz"]
    metadatas = [{"text": text} for text in texts]
    docsearch = Kinetica.from_texts(
        db=db,
        texts=texts,
        metadatas=metadatas,
        embedding=openai,
        dimensions=1536,
        collection_name="kinetica_openai_test",
        pre_delete_collection=False,
    )
    output = docsearch.similarity_search("foo", k=1)
    print(output)
    assert output[0].page_content == "foo"


if __name__ == "__main__":
    test_kinetica_max_marginal_relevance_search()

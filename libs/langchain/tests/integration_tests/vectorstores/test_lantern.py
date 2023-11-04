"""Test Lantern functionality."""
import os
from typing import List
import psycopg2

from sqlalchemy.orm import Session

from langchain.docstore.document import Document
from langchain.vectorstores import Lantern
from langchain.vectorstores.lantern import CollectionStore, EmbeddingStore
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

CONNECTION_STRING = Lantern.connection_string_from_db_params(
    driver=os.environ.get("TEST_LANTERN_DRIVER", "psycopg2"),
    host=os.environ.get("TEST_LANTERN_HOST", "localhost"),
    port=int(os.environ.get("TEST_LANTERN_PORT", "5432")),
    database=os.environ.get("TEST_LANTERN_DATABASE", "postgres"),
    user=os.environ.get("TEST_LANTERN_USER", "postgres"),
    password=os.environ.get("TEST_LANTERN_PASSWORD", "postgres"),
)


ADA_TOKEN_COUNT = 1536

def fix_distance_precision(results, precision = 2):
    return list(map(lambda x: (x[0], float(f"{{:.{precision}f}}".format(x[1]))), results))

class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]

def clean_database() -> None:
    """Clean database to avoid inconsistencies"""
    client = psycopg2.connect(CONNECTION_STRING.replace("+psycopg2",""))
    with client.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {EmbeddingStore.__tablename__} CASCADE")
        cur.execute(f"DROP TABLE IF EXISTS {CollectionStore.__tablename__} CASCADE")
    client.commit()
    client.close()
    
def test_lantern() -> None:
    """Test end to end construction and search."""
    clean_database()
    texts = ["foo", "bar", "baz"]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_lantern_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = Lantern.from_embeddings(
        text_embeddings=text_embedding_pairs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_lantern_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_lantern_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = fix_distance_precision(docsearch.similarity_search_with_score("foo", k=1))
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_lantern_with_filter_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = fix_distance_precision(docsearch.similarity_search_with_score("foo", k=1, filter={"page": "0"}))
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_lantern_with_filter_distant_match() -> None:
    # clean_database()
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = fix_distance_precision(docsearch.similarity_search_with_score("foo", k=1, filter={"page": "2"}))
    assert output == [
        (Document(page_content="baz", metadata={"page": "2"}), 0.0)
    ]


def test_lantern_with_filter_no_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "5"})
    assert output == []


def test_lantern_collection_with_metadata() -> None:
    """Test end to end collection construction"""
    lantern = Lantern(
        collection_name="test_collection",
        collection_metadata={"foo": "bar"},
        embedding_function=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    session = Session(lantern.connect())
    collection = lantern.get_collection(session)
    if collection is None:
        assert False, "Expected a CollectionStore object but received None"
    else:
        assert collection.name == "test_collection"
        assert collection.cmetadata == {"foo": "bar"}


def test_lantern_with_filter_in_set() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = fix_distance_precision(docsearch.similarity_search_with_score(
        "foo", k=2, filter={"page": {"IN": ["0", "2"]}}
    ), 4)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 0.0),
        (Document(page_content="baz", metadata={"page": "2"}), 0.0013),
    ]


def test_lantern_delete_docs() -> None:
    """Add and delete documents."""
    clean_database()
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        ids=["1", "2", "3"],
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    docsearch.delete(["1", "2", "3"])
    output = docsearch.similarity_search("foo", k=3)
    assert output == []


def test_lantern_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    clean_database()
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    print("running function")

    output = fix_distance_precision(docsearch.similarity_search_with_relevance_scores("foo", k=3), 4)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9997),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9987),
    ]


def test_lantern_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.999},
    )
    output = retriever.get_relevant_documents("summer")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]


def test_lantern_retriever_search_threshold_custom_normalization_fn() -> None:
    clean_database()
    """Test searching with threshold and custom normalization function"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection_string=CONNECTION_STRING,
        relevance_score_fn=lambda d: d * 0,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.9999},
    )
    output = retriever.get_relevant_documents("foo")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
    ]


def test_lantern_max_marginal_relevance_search() -> None:
    """Test max marginal relevance search."""
    texts = ["foo", "bar", "baz"]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)
    assert output == [Document(page_content="foo")]


def test_lantern_max_marginal_relevance_search_with_score() -> None:
    """Test max marginal relevance search with relevance scores."""
    texts = ["foo", "bar", "baz"]
    docsearch = Lantern.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = fix_distance_precision(docsearch.max_marginal_relevance_search_with_score("foo", k=1, fetch_k=3))
    assert output == [(Document(page_content="foo"), 0.0)]

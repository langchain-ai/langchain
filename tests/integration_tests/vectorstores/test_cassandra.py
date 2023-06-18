"""Test Cassandra functionality."""
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.vectorstores import Cassandra
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)

from cassandra.cluster import Cluster, Session

def _vectorstore_from_texts(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    drop: bool = True
) -> Cassandra:
    keyspace = 'vector_test_keyspace'
    table_name = 'vector_test_table'
    # get db connection
    cluster = Cluster()
    session = cluster.connect()
    # ensure keyspace exists
    session.execute(f"CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}")
    # drop table if required
    if drop:
        session.execute(f'DROP TABLE IF EXISTS {keyspace}.{table_name}')
    #
    return Cassandra.from_texts(
        texts,
        ConsistentFakeEmbeddings(),
        metadatas=metadatas,
        session=session,
        keyspace=keyspace,
        table_name=table_name,
    )


def test_cassandra() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = _vectorstore_from_texts(texts)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_cassandra_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(texts, metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]
    assert scores[0] > scores[1] > scores[2]


def test_cassandra_max_marginal_relevance_search() -> None:
    """Test end to end construction and MMR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(texts, metadatas=metadatas)
    output = docsearch.max_marginal_relevance_search("doh", k=2, fetch_k=3)
    assert output == [
        Document(page_content="baz", metadata={"page": 2}),
        Document(page_content="bar", metadata={"page": 1}),
    ]


def test_cassandra_add_extra() -> None:
    """Test end to end construction with further insertions."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(texts, metadatas=metadatas)

    docsearch.add_texts(texts, metadatas)
    texts2 = ["foo2", "bar2", "baz2"]
    docsearch.add_texts(texts2, metadatas)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


def test_cassandra_no_drop() -> None:
    """Test end to end construction and re-opening the same index."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(texts, metadatas=metadatas)
    del docsearch

    texts2 = ["foo2", "bar2", "baz2"]
    docsearch = _vectorstore_from_texts(texts2, metadatas=metadatas, drop=False)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


# if __name__ == "__main__":
#     test_cassandra()
#     test_cassandra_with_score()
#     test_cassandra_max_marginal_relevance_search()
#     test_cassandra_add_extra()
#     test_cassandra_no_drop()

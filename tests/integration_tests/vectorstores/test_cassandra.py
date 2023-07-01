"""Test Cassandra functionality."""
from typing import List, Optional, Type

from cassandra.cluster import Cluster

from langchain.docstore.document import Document
from langchain.vectorstores import Cassandra
from tests.integration_tests.vectorstores.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
    ConsistentFakeEmbeddings,
    Embeddings,
)


def _vectorstore_from_texts(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    embedding_class: Type[Embeddings] = ConsistentFakeEmbeddings,
    drop: bool = True,
) -> Cassandra:
    keyspace = "vector_test_keyspace"
    table_name = "vector_test_table"
    # get db connection
    cluster = Cluster()
    session = cluster.connect()
    # ensure keyspace exists
    session.execute(
        (
            f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
            f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
        )
    )
    # drop table if required
    if drop:
        session.execute(f"DROP TABLE IF EXISTS {keyspace}.{table_name}")
    #
    return Cassandra.from_texts(
        texts,
        embedding_class(),
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
    """
    Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        \  v1
    v3  |     .    | query
         \        /  v0
          \______/                 (N.B. very crude drawing)

    With fetch_k==3 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order).
    """
    texts = ["-0.125", "+0.125", "+0.25", "+1.0"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _vectorstore_from_texts(
        texts, metadatas=metadatas, embedding_class=AngularTwoDimensionalEmbeddings
    )
    output = docsearch.max_marginal_relevance_search("0.0", k=2, fetch_k=3)
    output_set = {
        (mmr_doc.page_content, mmr_doc.metadata["page"]) for mmr_doc in output
    }
    assert output_set == {
        ("+0.25", 2),
        ("-0.125", 0),
    }


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

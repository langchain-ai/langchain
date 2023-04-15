"""Test Milvus functionality."""
import os
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.vectorstores import Zilliz
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


collection_name = "langchain_test_index"  # name of the index


def _zilliz_from_texts(metadatas: Optional[List[dict]] = None) -> Zilliz:
    return Zilliz.from_texts(
        fake_texts,
        FakeEmbeddings(),
        collection_name=collection_name,
        metadatas=metadatas,
        connection_args={
          "host": os.environ.get("ZILLIZ_CLOUD_HOST"), 
          "port": os.environ.get("ZILLIZ_CLOUD_PORT"),
          "secure": True,
          "user": os.environ.get("ZILLIZ_CLOUD_USER"),
          "password": os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        },
    )


def test_zilliz() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _zilliz_from_texts(metadatas)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": 0})]


def test_zilliz_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _zilliz_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]
    assert scores[0] < scores[1] < scores[2]


def test_zilliz_max_marginal_relevance_search() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _zilliz_from_texts(metadatas=metadatas)
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="baz", metadata={"page": 2}),
    ]

def test_zilliz_from_existing_collection() -> None:
    """Test that namespaces are properly handled."""
    # Create two indexes with the same name but different partitions
    texts_1 = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts_1))]
    Zilliz.from_texts(
        texts_1,
        FakeEmbeddings(),
        collection_name=collection_name,
        metadatas=metadatas,
        connection_args={
          "host": os.environ.get("ZILLIZ_CLOUD_HOST"), 
          "port": os.environ.get("ZILLIZ_CLOUD_PORT"),
          "secure": True,
          "user": os.environ.get("ZILLIZ_CLOUD_USER"),
          "password": os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        },
    )

    # Search
    docsearch = Zilliz.from_existing_collectyion(
        collection_name=collection_name,
        embedding=FakeEmbeddings(),
        connection_args={
          "host": os.environ.get("ZILLIZ_CLOUD_HOST"), 
          "port": os.environ.get("ZILLIZ_CLOUD_PORT"),
          "secure": True,
          "user": os.environ.get("ZILLIZ_CLOUD_USER"),
          "password": os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        },
    )
    output = docsearch.similarity_search("foo", k=20)
    # check that we don't get results from the other namespace
    page_contents = sorted(set([o.page_content for o in output]))
    assert all(content in ["foo", "bar", "baz"] for content in page_contents)
    assert all(content not in ["foo2", "bar2", "baz2"] for content in page_contents)


# Zilliz Cloud has not yet implemented the Partition function so far, it may take some time. will enable this unit test once the Partition function is available.
def __test_zilliz_from_existing_collection_with_partition(embedding_openai: FakeEmbeddings) -> None:
    """Test that namespaces are properly handled."""
    # Create two indexes with the same name but different partitions
    texts_1 = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts_1))]
    Zilliz.from_texts(
        texts_1,
        FakeEmbeddings(),
        collection_name=collection_name,
        partition_name=f"{collection_name}_1",
        metadatas=metadatas,
        connection_args={
          "host": os.environ.get("ZILLIZ_CLOUD_HOST"), 
          "port": os.environ.get("ZILLIZ_CLOUD_PORT"),
          "secure": True,
          "user": os.environ.get("ZILLIZ_CLOUD_USER"),
          "password": os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        },
    )


    texts_2 = ["foo2", "bar2", "baz2"]
    metadatas = [{"page": i} for i in range(len(texts_2))]
    Zilliz.from_texts(
        texts_2,
        FakeEmbeddings(),
        collection_name=collection_name,
        partition_name=f"{collection_name}_2",
        metadatas=metadatas,
        connection_args={
          "host": os.environ.get("ZILLIZ_CLOUD_HOST"), 
          "port": os.environ.get("ZILLIZ_CLOUD_PORT"),
          "secure": True,
          "user": os.environ.get("ZILLIZ_CLOUD_USER"),
          "password": os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        },
    )

    # Search
    docsearch = Zilliz.from_existing_collectyion(
        collection_name=collection_name,
        embedding=embedding_openai,
        connection_args={
          "host": os.environ.get("ZILLIZ_CLOUD_HOST"), 
          "port": os.environ.get("ZILLIZ_CLOUD_PORT"),
          "secure": True,
          "user": os.environ.get("ZILLIZ_CLOUD_USER"),
          "password": os.environ.get("ZILLIZ_CLOUD_PASSWORD"),
        },
    )
    output = docsearch.similarity_search("foo", k=20, partition_names=[f"{collection_name}_1"])
    # check that we don't get results from the other namespace
    page_contents = sorted(set([o.page_content for o in output]))
    assert all(content in ["foo", "bar", "baz"] for content in page_contents)
    assert all(content not in ["foo2", "bar2", "baz2"] for content in page_contents)

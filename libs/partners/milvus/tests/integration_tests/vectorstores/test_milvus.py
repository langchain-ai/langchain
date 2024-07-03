"""Test Milvus functionality."""
from typing import Any, List, Optional

from langchain_core.documents import Document

from langchain_milvus.vectorstores import Milvus
from tests.integration_tests.utils import (
    FakeEmbeddings,
    assert_docs_equal_without_pk,
    fake_texts,
)

#
# To run this test properly, please start a Milvus server with the following command:
#
# ```shell
# wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
# bash standalone_embed.sh start
# ```
#
# Here is the reference:
# https://milvus.io/docs/install_standalone-docker.md
#


def _milvus_from_texts(
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    drop: bool = True,
) -> Milvus:
    return Milvus.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
        # connection_args={"uri": "http://127.0.0.1:19530"},
        connection_args={"uri": "./milvus_demo.db"},
        drop_old=drop,
    )


def _get_pks(expr: str, docsearch: Milvus) -> List[Any]:
    return docsearch.get_pks(expr)  # type: ignore[return-value]


def test_milvus() -> None:
    """Test end to end construction and search."""
    docsearch = _milvus_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert_docs_equal_without_pk(output, [Document(page_content="foo")])


def test_milvus_with_metadata() -> None:
    """Test with metadata"""
    docsearch = _milvus_from_texts(metadatas=[{"label": "test"}] * len(fake_texts))
    output = docsearch.similarity_search("foo", k=1)
    assert_docs_equal_without_pk(
        output, [Document(page_content="foo", metadata={"label": "test"})]
    )


def test_milvus_with_id() -> None:
    """Test with ids"""
    ids = ["id_" + str(i) for i in range(len(fake_texts))]
    docsearch = _milvus_from_texts(ids=ids)
    output = docsearch.similarity_search("foo", k=1)
    assert_docs_equal_without_pk(output, [Document(page_content="foo")])

    output = docsearch.delete(ids=ids)
    assert output.delete_count == len(fake_texts)  # type: ignore[attr-defined]

    try:
        ids = ["dup_id" for _ in fake_texts]
        _milvus_from_texts(ids=ids)
    except Exception as e:
        assert isinstance(e, AssertionError)


def test_milvus_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert_docs_equal_without_pk(
        docs,
        [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
            Document(page_content="baz", metadata={"page": 2}),
        ],
    )
    assert scores[0] < scores[1] < scores[2]


def test_milvus_max_marginal_relevance_search() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_texts(metadatas=metadatas)
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert_docs_equal_without_pk(
        output,
        [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="baz", metadata={"page": 2}),
        ],
    )


def test_milvus_add_extra() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_texts(metadatas=metadatas)

    docsearch.add_texts(texts, metadatas)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


def test_milvus_no_drop() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _milvus_from_texts(metadatas=metadatas)
    del docsearch

    docsearch = _milvus_from_texts(metadatas=metadatas, drop=False)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


def test_milvus_get_pks() -> None:
    """Test end to end construction and get pks with expr"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"id": i} for i in range(len(texts))]
    docsearch = _milvus_from_texts(metadatas=metadatas)
    expr = "id in [1,2]"
    output = _get_pks(expr, docsearch)
    assert len(output) == 2


def test_milvus_delete_entities() -> None:
    """Test end to end construction and delete entities"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"id": i} for i in range(len(texts))]
    docsearch = _milvus_from_texts(metadatas=metadatas)
    expr = "id in [1,2]"
    pks = _get_pks(expr, docsearch)
    result = docsearch.delete(pks)
    assert result.delete_count == 2  # type: ignore[attr-defined]


def test_milvus_upsert_entities() -> None:
    """Test end to end construction and upsert entities"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"id": i} for i in range(len(texts))]
    docsearch = _milvus_from_texts(metadatas=metadatas)
    expr = "id in [1,2]"
    pks = _get_pks(expr, docsearch)
    documents = [
        Document(page_content="test_1", metadata={"id": 1}),
        Document(page_content="test_2", metadata={"id": 3}),
    ]
    ids = docsearch.upsert(pks, documents)
    assert len(ids) == 2  # type: ignore[arg-type]


# if __name__ == "__main__":
#     test_milvus()
#     test_milvus_with_metadata()
#     test_milvus_with_id()
#     test_milvus_with_score()
#     test_milvus_max_marginal_relevance_search()
#     test_milvus_add_extra()
#     test_milvus_no_drop()
#     test_milvus_get_pks()
#     test_milvus_delete_entities()
#     test_milvus_upsert_entities()

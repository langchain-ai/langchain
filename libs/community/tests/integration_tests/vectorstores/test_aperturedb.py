"""Test ApertureDB functionality."""

import uuid
from typing import List, Optional

from langchain_core.documents import Document

from langchain_community.vectorstores import ApertureDB
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _aperturedb_from_texts(
    metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None
) -> ApertureDB:
    """Create an ApertureDB instance from fake texts."""
    descriptor_set = uuid.uuid4().hex  # Fresh descriptor set for each test
    if ids is None:
        return ApertureDB.from_texts(
            fake_texts,
            FakeEmbeddings(),
            metadatas=metadatas,
            descriptor_set=descriptor_set,
        )
    else:  # to supply ids, we have to use the Document class
        if metadatas is None:
            docs = [
                Document(page_content=text, id=id_)
                for text, id_ in zip(fake_texts, ids)
            ]
        else:
            docs = [
                Document(page_content=text, id=id_, metadata=metadata)
                for text, id_, metadata in zip(fake_texts, ids, metadatas)
            ]
        print("ID", docs[0].id)
        print(docs)
        return ApertureDB.from_documents(
            docs,
            FakeEmbeddings(),
            descriptor_set=descriptor_set,
        )


def test_aperturedb() -> None:
    """Test end to end construction and search."""
    docsearch = _aperturedb_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_aperturedb_with_metadata() -> None:
    """Test with metadata"""
    docsearch = _aperturedb_from_texts(metadatas=[{"label": "test"}] * len(fake_texts))
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"label": "test"})]


def test_aperturedb_with_id() -> None:
    """Test with ids"""
    ids = ["id_" + str(i) for i in range(len(fake_texts))]
    docsearch = _aperturedb_from_texts(ids=ids)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]
    assert output[0].id == "id_0"

    output = docsearch.delete(ids=ids)
    assert output

    output = docsearch.similarity_search("foo", k=1)
    assert output == []


def test_aperturedb_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    metadatas = [{"page": i} for i in range(len(fake_texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ], docs
    # scores descending by default (CS)
    assert scores[0] > scores[1] > scores[2], scores


def test_aperturedb_max_marginal_relevance_search() -> None:
    """Test end to end construction and MRR search."""
    metadatas = [{"page": i} for i in range(len(fake_texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    assert output == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="baz", metadata={"page": 2}),
    ], output


def test_aperturedb_add_extra() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)

    docsearch.add_texts(texts, metadatas)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6, len(output)


def test_aperturedb_no_drop() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)
    del docsearch

    docsearch = _aperturedb_from_texts(metadatas=metadatas, drop=False)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6, len(output)


def test_aperturedb_delete_entities() -> None:
    """Test end to end construction and delete entities"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"id": i} for i in range(len(texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)
    expr = "id in [1,2]"
    pks = _get_pks(expr, docsearch)
    result = docsearch.delete(pks)
    assert result is True


def test_aperturedb_upsert_entities() -> None:
    """Test end to end construction and upsert entities"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"id": i} for i in range(len(texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)
    expr = "id in [1,2]"
    pks = _get_pks(expr, docsearch)
    documents = [
        Document(page_content="test_1", metadata={"id": 1}),
        Document(page_content="test_2", metadata={"id": 3}),
    ]
    ids = docsearch.upsert(pks, documents)
    assert len(ids) == 2, len(ids)  # type: ignore[arg-type]


if __name__ == "__main__":
    test_aperturedb()
    test_aperturedb_with_metadata()
    test_aperturedb_with_id()
    test_aperturedb_with_score()
    test_aperturedb_max_marginal_relevance_search()
    test_aperturedb_add_extra()
    test_aperturedb_no_drop()

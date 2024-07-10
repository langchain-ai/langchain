"""Test ApertureDB functionality."""

import uuid
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.indexing.base import UpsertResponse

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
    assert metadatas is None or len(metadatas) == len(fake_texts)
    assert ids is None or len(ids) == len(fake_texts)
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
        return ApertureDB.from_documents(
            docs,
            FakeEmbeddings(),
            descriptor_set=descriptor_set,
        )


def _compare_documents(actuals: List[Document], expecteds: List[Document]) -> None:
    """Compare two documents, with one-sided test on IDs.
    If we don't provide an ID, one will be generated for us, and we don't care what
    it is."""
    assert len(actuals) == len(
        expecteds
    ), f"Expected {expecteds} results, got {actuals}"
    for i, (actual, expected) in enumerate(zip(actuals, expecteds)):
        assert (
            actual.page_content == expected.page_content
        ), f"{i}: page_content {actual.page_content} != {expected.page_content}"
        assert (
            actual.metadata == expected.metadata
        ), f"{i}: metadata {actual.metadata} != {expected.metadata}"
        if expected.id is not None:
            assert actual.id == expected.id, f"{i}: id {actual.id} != {expected.id}"


def test_aperturedb() -> None:
    """Test end to end construction and search."""
    docsearch = _aperturedb_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    _compare_documents(output, [Document(page_content="foo")])
    ApertureDB.delete_vectorstore(docsearch.descriptor_set)


def test_aperturedb_with_metadata() -> None:
    """Test with metadata"""
    docsearch = _aperturedb_from_texts(
        metadatas=[{"label": "test"}] * len(fake_texts))
    output = docsearch.similarity_search("foo", k=1)
    _compare_documents(
        output, [Document(page_content="foo", metadata={"label": "test"})]
    )
    ApertureDB.delete_vectorstore(docsearch.descriptor_set)


def test_aperturedb_with_id() -> None:
    """Test with ids"""
    ids = ["id_" + str(i) for i in range(len(fake_texts))]
    docsearch = _aperturedb_from_texts(ids=ids)
    output = docsearch.similarity_search("foo", k=1)
    _compare_documents(output, [Document(page_content="foo", id="id_0")])

    output = docsearch.delete(ids=ids)
    assert output

    output = docsearch.similarity_search("foo", k=1)
    assert output == []

    ApertureDB.delete_vectorstore(docsearch.descriptor_set)


def test_aperturedb_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    metadatas = [{"page": i} for i in range(len(fake_texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    _compare_documents(
        docs,
        [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
            Document(page_content="baz", metadata={"page": 2}),
        ],
    )
    # scores descending by default (CS)
    assert (
        scores[0] > scores[1] > scores[2]
    ), f"Expected {scores[0]} > {scores[1]} > {scores[2]}"
    ApertureDB.delete_vectorstore(docsearch.descriptor_set)


def test_aperturedb_max_marginal_relevance_search() -> None:
    """Test end to end construction and MRR search."""
    metadatas = [{"page": i} for i in range(len(fake_texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)
    output = docsearch.max_marginal_relevance_search("foo", k=2, fetch_k=3)
    _compare_documents(
        output,
        [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
        ],
    )
    ApertureDB.delete_vectorstore(docsearch.descriptor_set)


def test_aperturedb_add_extra() -> None:
    """Test end to end construction and similarity search."""
    texts = fake_texts
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)

    docsearch.add_texts(texts, metadatas)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6, len(output)
    ApertureDB.delete_vectorstore(docsearch.descriptor_set)


def test_aperturedb_add_extra_mmr() -> None:
    """Test end to end construction and MRR search."""
    texts = fake_texts
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _aperturedb_from_texts(metadatas=metadatas)

    docsearch.add_texts(texts, metadatas)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6, len(output)
    ApertureDB.delete_vectorstore(docsearch.descriptor_set)


def test_aperturedb_generate_ids() -> None:
    """Test that the generated ids work correctly"""
    descriptor_set = uuid.uuid4().hex  # Fresh descriptor set for each test
    vectorstore = ApertureDB(embeddings=FakeEmbeddings(),
                             descriptor_set=descriptor_set)
    ids = vectorstore.add_texts(fake_texts)
    docs = vectorstore.similarity_search("foo", k=10)
    ids2 = [doc.id for doc in docs]
    assert set(ids) == set(ids2), (ids, ids2)
    ApertureDB.delete_vectorstore(vectorstore.descriptor_set)


def test_aperturedb_upsert_entities() -> None:
    """Test end to end construction and upsert entities"""
    ids = range(len(fake_texts))
    vectorstore = _aperturedb_from_texts(ids=ids)
    documents = [
        Document(page_content="test1", id=1),
        Document(page_content="test2", id=100),
    ]
    response = vectorstore.upsert(documents)
    assert response.succeeded == [1, 100], response.succeeded
    assert response.failed == [], response.failed
    docs = vectorstore.similarity_search("foo", k=10)
    ids2 = [doc.id for doc in docs]
    assert set(ids) == set(ids) + {100}, (ids, ids2)
    docs_by_id = {doc.id: doc for doc in docs}
    assert docs_by_id[1].page_content == "test1"
    assert docs_by_id[100].page_content == "test2"
    ApertureDB.delete_vectorstore(vectorstore.descriptor_set)


if __name__ == "__main__":
    test_aperturedb()
    test_aperturedb_with_metadata()
    test_aperturedb_with_id()
    test_aperturedb_with_score()
    test_aperturedb_max_marginal_relevance_search()
    test_aperturedb_add_extra()
    test_aperturedb_add_extra_mmr()
    test_aperturedb_generate_ids()
    test_aperturedb_upsert_entities()
    print("All tests passed")

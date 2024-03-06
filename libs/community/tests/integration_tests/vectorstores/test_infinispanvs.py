"""Test Infinispan functionality."""
from typing import Any, List, Optional

from langchain_core.documents import Document

from langchain_community.vectorstores import InfinispanVS
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _infinispan_setup() -> None:
    ispnvs = InfinispanVS()
    ispnvs.cache_delete()
    ispnvs.schema_delete()
    proto = """
    /**
     * @Indexed
     */
    message vector {
    /**
     * @Vector(dimension=10)
     */
    repeated float vector = 1;
    optional string text = 2;
    optional string label = 3;
    optional int32 page = 4;
    }
    """
    ispnvs.schema_create(proto)
    ispnvs.cache_create()
    ispnvs.cache_index_clear()


def _infinispanvs_from_texts(
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
    clear_old: Optional[bool] = True,
    **kwargs: Any,
) -> InfinispanVS:
    texts = [{"text": t} for t in fake_texts]
    if metadatas is None:
        metadatas = texts
    else:
        [m.update(t) for (m, t) in zip(metadatas, texts)]
    return InfinispanVS.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        ids=ids,
        clear_old=clear_old,
        **kwargs,
    )


def test_infinispan() -> None:
    """Test end to end construction and search."""
    _infinispan_setup()
    docsearch = _infinispanvs_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_infinispan_with_metadata() -> None:
    """Test with metadata"""
    _infinispan_setup()
    meta = []
    for _ in range(len(fake_texts)):
        meta.append({"label": "test"})
    docsearch = _infinispanvs_from_texts(metadatas=meta)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"label": "test"})]


def test_infinispan_with_metadata_with_output_fields() -> None:
    """Test with metadata"""
    _infinispan_setup()
    metadatas = [{"page": i, "label": "label" + str(i)} for i in range(len(fake_texts))]
    c = {"output_fields": ["label", "page", "text"]}
    docsearch = _infinispanvs_from_texts(metadatas=metadatas, configuration=c)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [
        Document(page_content="foo", metadata={"label": "label0", "page": 0})
    ]


def test_infinispanvs_with_id() -> None:
    """Test with ids"""
    ids = ["id_" + str(i) for i in range(len(fake_texts))]
    docsearch = _infinispanvs_from_texts(ids=ids)
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_infinispan_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    _infinispan_setup()
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _infinispanvs_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": 0}),
        Document(page_content="bar", metadata={"page": 1}),
        Document(page_content="baz", metadata={"page": 2}),
    ]
    assert scores[0] >= scores[1] >= scores[2]


def test_infinispan_add_texts() -> None:
    """Test end to end construction and MRR search."""
    _infinispan_setup()
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _infinispanvs_from_texts(metadatas=metadatas)

    docsearch.add_texts(texts, metadatas)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


def test_infinispan_no_clear_old() -> None:
    """Test end to end construction and MRR search."""
    _infinispan_setup()
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _infinispanvs_from_texts(metadatas=metadatas)
    del docsearch
    docsearch = _infinispanvs_from_texts(metadatas=metadatas, clear_old=False)
    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6

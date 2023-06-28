"""Test Zilliz functionality."""
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.vectorstores import Zilliz
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _zilliz_from_texts(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> Zilliz:
    return Zilliz.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        connection_args={
            "uri": "",
            "user": "",
            "password": "",
            "secure": True,
        },
        drop_old=drop,
    )


def test_zilliz() -> None:
    """Test end to end construction and search."""
    docsearch = _zilliz_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


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


def test_zilliz_add_extra() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _zilliz_from_texts(metadatas=metadatas)

    docsearch.add_texts(texts, metadatas)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


def test_zilliz_no_drop() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _zilliz_from_texts(metadatas=metadatas)
    del docsearch

    docsearch = _zilliz_from_texts(metadatas=metadatas, drop=False)

    output = docsearch.similarity_search("foo", k=10)
    assert len(output) == 6


# if __name__ == "__main__":
#     test_zilliz()
#     test_zilliz_with_score()
#     test_zilliz_max_marginal_relevance_search()
#     test_zilliz_add_extra()
#     test_zilliz_no_drop()

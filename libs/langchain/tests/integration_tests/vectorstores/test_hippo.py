"""Test Hippo functionality."""
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.vectorstores.hippo import Hippo
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _hippo_from_texts(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> Hippo:
    return Hippo.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        table_name="langchain_test",
        connection_args={"host": "127.0.0.1", "port": 7788},
        drop_old=drop,
    )


def test_hippo_add_extra() -> None:
    """Test end to end construction and MRR search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _hippo_from_texts(metadatas=metadatas)

    docsearch.add_texts(texts, metadatas)

    output = docsearch.similarity_search("foo", k=1)
    print(output)
    assert len(output) == 1


def test_hippo() -> None:
    docsearch = _hippo_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_hippo_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": i} for i in range(len(texts))]
    docsearch = _hippo_from_texts(metadatas=metadatas)
    output = docsearch.similarity_search_with_score("foo", k=3)
    docs = [o[0] for o in output]
    scores = [o[1] for o in output]
    assert docs == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
        Document(page_content="baz", metadata={"page": "2"}),
    ]
    assert scores[0] < scores[1] < scores[2]


# if __name__ == "__main__":
# test_hippo()
# test_hippo_with_score()
# test_hippo_with_score()

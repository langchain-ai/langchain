"""Test BESVectorStore functionality."""
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.vectorstores import BESVectorStore
from tests.integration_tests.vectorstores.fake_embeddings import (
    FakeEmbeddings,
    fake_texts,
)


def _bes_vector_db_from_texts(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> BESVectorStore:
    return BESVectorStore.from_texts(
        fake_texts,
        FakeEmbeddings(),
        metadatas=metadatas,
        bes_url="http://10.0.X.X",
    )


def test_bes_vector_db() -> None:
    """Test end to end construction and search."""
    docsearch = _bes_vector_db_from_texts()
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]

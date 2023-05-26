from typing import List

from langchain.document_loaders import GitHubLoader
from langchain.schema import Document


def assert_docs(docs: List[Document]) -> None:
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        assert set(doc.metadata) == {"url", "title", "creator", "creation_time", "comments"}


def test_load_success() -> None:
    """Test that returns one document"""
    loader = GitHubLoader(query="1605.08386", load_max_docs=2)

    docs = loader.load()
    assert len(docs) == 1
    print(docs[0].metadata)
    print(docs[0].page_content)
    assert_docs(docs)


def test_failure() -> None:
    """Test that returns no docs"""
    loader = GitHubLoader(load_max_docs=2)
    docs = loader.load()

    assert len(docs) == 0

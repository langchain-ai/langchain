from typing import Any, Callable

import pytest
from pytest import MonkeyPatch

from langchain.docstore.document import Document
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader


@pytest.fixture
def url_loader() -> RecursiveUrlLoader:
    url = "http://test.com"
    exclude_dir = "/exclude"  # Note: Changed from list to single string
    return RecursiveUrlLoader(url, exclude_dir)


@pytest.fixture
def mock_web_base_loader_load(monkeypatch: MonkeyPatch) -> None:
    """Mock requests.get"""

    # Mock Document object for relative URL
    mock_doc_relative = Document(
        page_content="Relative page", metadata={"url": "http://test.com/relative"}
    )
    # Mock Document object for absolute URL
    mock_doc_absolute = Document(
        page_content="Absolute page", metadata={"url": "http://test.com/absolute"}
    )

    def mock_get(url: str, *args: Any, **kwargs: Any) -> Document:
        if url.startswith("http://test.com"):
            if "/absolute" in url:
                return mock_doc_absolute
            elif "/relative" in url:
                return mock_doc_relative
        default_doc = Document(
            page_content="Default page", metadata={"url": "http://test.com/default"}
        )
        return default_doc

    monkeypatch.setattr("langchain.document_loaders.WebBaseLoader.load", mock_get)


def test_get_child_links_recursive(
    url_loader: RecursiveUrlLoader, mock_web_base_loader_load: Callable[[], None]
) -> None:
    # Testing for both relative and absolute URL
    child_docs = list(url_loader.get_child_links_recursive("http://test.com"))

    assert len(child_docs) == 2
    assert set(doc.metadata["url"] for doc in child_docs) == {
        "http://test.com/relative",
        "http://test.com/absolute",
    }

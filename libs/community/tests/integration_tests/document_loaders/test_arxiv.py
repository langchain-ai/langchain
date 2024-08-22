import shutil
from http.client import HTTPMessage
from pathlib import Path
from typing import List, Union
from unittest.mock import patch
from urllib.error import HTTPError

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders.arxiv import ArxivLoader

EXAMPLE_HELLO_PDF_PATH = Path(__file__).parents[1] / "examples" / "hello.pdf"


def assert_docs(docs: List[Document]) -> None:
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        assert set(doc.metadata) == {"Published", "Title", "Authors", "Summary"}


def test_load_success() -> None:
    """Test that returns one document"""
    loader = ArxivLoader(query="1605.08386", load_max_docs=2)

    docs = loader.load()
    assert len(docs) == 1
    print(docs[0].metadata)  # noqa: T201
    print(docs[0].page_content)  # noqa: T201
    assert_docs(docs)


def test_load_returns_no_result() -> None:
    """Test that returns no docs"""
    loader = ArxivLoader(query="1605.08386WWW", load_max_docs=2)
    docs = loader.load()

    assert len(docs) == 0


def test_load_returns_limited_docs() -> None:
    """Test that returns several docs"""
    expected_docs = 2
    loader = ArxivLoader(query="ChatGPT", load_max_docs=expected_docs)
    docs = loader.load()

    assert len(docs) == expected_docs
    assert_docs(docs)


def test_load_returns_full_set_of_metadata() -> None:
    """Test that returns several docs"""
    loader = ArxivLoader(query="ChatGPT", load_max_docs=1, load_all_available_meta=True)
    docs = loader.load()
    assert len(docs) == 1
    for doc in docs:
        assert doc.page_content
        assert doc.metadata
        assert set(doc.metadata).issuperset(
            {"Published", "Title", "Authors", "Summary"}
        )
        print(doc.metadata)  # noqa: T201
        assert len(set(doc.metadata)) > 4


def test_skip_http_error() -> None:
    """Test skipping unexpected Http 404 error of a single doc"""
    tmp_hello_pdf_path = Path(__file__).parent / "hello.pdf"

    def first_download_fails() -> Union[HTTPError, str]:
        if not hasattr(first_download_fails, "firstCall"):
            first_download_fails.__setattr__("firstCall", False)
            raise HTTPError(
                url="", code=404, msg="Not Found", hdrs=HTTPMessage(), fp=None
            )
        else:
            # Return temporary example pdf path
            shutil.copy(EXAMPLE_HELLO_PDF_PATH, tmp_hello_pdf_path)
            return str(tmp_hello_pdf_path.absolute())

    with patch("arxiv.Result.download_pdf") as mock_download_pdf:
        # Set up the mock to raise HTTP 404 error
        mock_download_pdf.side_effect = first_download_fails
        # Load documents
        loader = ArxivLoader(
            query="ChatGPT",
            load_max_docs=2,
            load_all_available_meta=True,
            continue_on_failure=True,
        )
        docs = loader.load()
        # Only 1 of 2 documents should be loaded
        assert len(docs) == 1


@pytest.mark.skip(reason="test could be flaky")
def test_load_issue_9046() -> None:
    """Test for the fixed issue 9046"""
    expected_docs = 3

    # ":" character could not be an issue
    loader = ArxivLoader(
        query="MetaGPT: Meta Programming for Multi-Agent Collaborative Framework",
        load_max_docs=expected_docs,
    )
    docs = loader.load()

    assert_docs(docs)
    assert "MetaGPT" in docs[0].metadata["Title"]

    # "-" character could not be an issue
    loader = ArxivLoader(
        query="MetaGPT - Meta Programming for Multi-Agent Collaborative Framework",
        load_max_docs=expected_docs,
    )
    docs = loader.load()

    assert_docs(docs)
    assert "MetaGPT" in docs[0].metadata["Title"]

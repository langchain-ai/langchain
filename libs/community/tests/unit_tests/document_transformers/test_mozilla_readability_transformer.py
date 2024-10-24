"""Unit tests for mozilla readability document transformer."""

import pytest
from langchain_core.documents import Document

from langchain_community.document_transformers import ReadabilityTransformer


@pytest.mark.requires("python-readability")
def test_empty_html() -> None:
    readability = ReadabilityTransformer()
    documents = [Document(page_content="<html></html>")]
    [document] = readability.transform_documents(documents)
    assert document.page_content == "", document.page_content


@pytest.mark.requires("python-readability")
def test_transform_non_empty_html() -> None:
    readability = ReadabilityTransformer()
    documents = [Document(page_content="<html><body><div>123</div></body></html>")]
    [document] = readability.transform_documents(documents)
    assert document.page_content == "123"

    [document] = readability.transform_documents(documents, target="html")
    assert (
        document.page_content
        == '<div id="readability-page-1" class="page"><p>123</p></div>'
    ), document.page_content

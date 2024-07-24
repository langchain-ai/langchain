"""Unit tests for html2text document transformer."""

import pytest
from langchain_core.documents import Document

from langchain_community.document_transformers import Html2TextTransformer


@pytest.mark.requires("html2text")
def test_transform_empty_html() -> None:
    html2text_transformer = Html2TextTransformer()
    empty_html = "<html></html>"
    documents = [Document(page_content=empty_html)]
    docs_transformed = html2text_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == "\n\n"


@pytest.mark.requires("html2text")
def test_extract_paragraphs() -> None:
    html2text_transformer = Html2TextTransformer()
    paragraphs_html = (
        "<html><h1>Header</h1><p>First paragraph.</p>"
        "<p>Second paragraph.</p><h1>Ignore at end</h1></html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = html2text_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# Header\n\n"
        "First paragraph.\n\n"
        "Second paragraph.\n\n"
        "# Ignore at end\n\n"
    )


@pytest.mark.requires("html2text")
def test_extract_html() -> None:
    html2text_transformer = Html2TextTransformer()
    paragraphs_html = (
        "<html>Begin of html tag"
        "<h1>Header</h1>"
        "<p>First paragraph.</p>"
        "Middle of html tag"
        "<p>Second paragraph.</p>"
        "End of html tag"
        "</html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = html2text_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "Begin of html tag\n\n"
        "# Header\n\n"
        "First paragraph.\n\n"
        "Middle of html tag\n\n"
        "Second paragraph.\n\n"
        "End of html tag\n\n"
    )


@pytest.mark.requires("html2text")
def test_remove_style() -> None:
    html2text_transformer = Html2TextTransformer()
    with_style_html = (
        "<html><style>my_funky_style</style><p>First paragraph.</p></html>"
    )
    documents = [Document(page_content=with_style_html)]
    docs_transformed = html2text_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == "First paragraph.\n\n"


@pytest.mark.requires("html2text")
def test_ignore_links() -> None:
    html2text_transformer = Html2TextTransformer(ignore_links=False)
    multiple_tags_html = (
        "<h1>First heading.</h1>"
        "<p>First paragraph with an <a href='http://example.com'>example</a></p>"
    )
    documents = [Document(page_content=multiple_tags_html)]

    docs_transformed = html2text_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# First heading.\n\n"
        "First paragraph with an [example](http://example.com)\n\n"
    )

    html2text_transformer = Html2TextTransformer(ignore_links=True)
    docs_transformed = html2text_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# First heading.\n\n" "First paragraph with an example\n\n"
    )


@pytest.mark.requires("html2text")
def test_ignore_images() -> None:
    html2text_transformer = Html2TextTransformer(ignore_images=False)
    multiple_tags_html = (
        "<h1>First heading.</h1>"
        "<p>First paragraph with an "
        "<img src='example.jpg' alt='Example image' width='500' height='600'></p>"
    )
    documents = [Document(page_content=multiple_tags_html)]

    docs_transformed = html2text_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# First heading.\n\n"
        "First paragraph with an ![Example image](example.jpg)\n\n"
    )

    html2text_transformer = Html2TextTransformer(ignore_images=True)
    docs_transformed = html2text_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# First heading.\n\n" "First paragraph with an\n\n"
    )

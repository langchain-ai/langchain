"""Unit tests for beautiful soup document transformer."""

import pytest
from langchain_core.documents import Document

from langchain_community.document_transformers import BeautifulSoupTransformer


@pytest.mark.requires("bs4")
def test_transform_empty_html() -> None:
    bs_transformer = BeautifulSoupTransformer()
    empty_html = "<html></html>"
    documents = [Document(page_content=empty_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == ""


@pytest.mark.requires("bs4")
def test_extract_paragraphs() -> None:
    bs_transformer = BeautifulSoupTransformer()
    paragraphs_html = (
        "<html><h1>Header</h1><p>First paragraph.</p>"
        "<p>Second paragraph.</p><h1>Ignore at end</h1></html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == "First paragraph. Second paragraph."


@pytest.mark.requires("bs4")
def test_strip_whitespace() -> None:
    bs_transformer = BeautifulSoupTransformer()
    paragraphs_html = (
        "<html><h1>Header</h1><p><span>First</span>   paragraph.</p>"
        "<p>Second paragraph. </p></html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == "First paragraph. Second paragraph."


@pytest.mark.requires("bs4")
def test_extract_html() -> None:
    bs_transformer = BeautifulSoupTransformer()
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
    docs_transformed = bs_transformer.transform_documents(
        documents, tags_to_extract=["html", "p"]
    )
    assert docs_transformed[0].page_content == (
        "Begin of html tag "
        "Header First paragraph. "
        "Middle of html tag "
        "Second paragraph. "
        "End of html tag"
    )


@pytest.mark.requires("bs4")
def test_remove_style() -> None:
    bs_transformer = BeautifulSoupTransformer()
    with_style_html = (
        "<html><style>my_funky_style</style><p>First paragraph.</p></html>"
    )
    documents = [Document(page_content=with_style_html)]
    docs_transformed = bs_transformer.transform_documents(
        documents, tags_to_extract=["html"]
    )
    assert docs_transformed[0].page_content == "First paragraph."


@pytest.mark.requires("bs4")
def test_remove_nested_tags() -> None:
    """
    If a tag_to_extract is _inside_ an unwanted_tag, it should be removed
    (e.g. a <p> inside a <table> if <table> is unwanted).)
    If an unwanted tag is _inside_ a tag_to_extract, it should be removed,
    but the rest of the tag_to_extract should stay.
    This means that "unwanted_tags" have a higher "priority" than "tags_to_extract".
    """
    bs_transformer = BeautifulSoupTransformer()
    with_style_html = (
        "<html><style>my_funky_style</style>"
        "<table><td><p>First paragraph, inside a table.</p></td></table>"
        "<p>Second paragraph<table><td> with a cell </td></table>.</p>"
        "</html>"
    )
    documents = [Document(page_content=with_style_html)]
    docs_transformed = bs_transformer.transform_documents(
        documents, unwanted_tags=["script", "style", "table"]
    )
    assert docs_transformed[0].page_content == "Second paragraph."


@pytest.mark.requires("bs4")
def test_remove_unwanted_lines() -> None:
    bs_transformer = BeautifulSoupTransformer()
    with_lines_html = "<html>\n\n<p>First \n\n     paragraph.</p>\n</html>\n\n"
    documents = [Document(page_content=with_lines_html)]
    docs_transformed = bs_transformer.transform_documents(documents, remove_lines=True)
    assert docs_transformed[0].page_content == "First paragraph."


@pytest.mark.requires("bs4")
def test_do_not_remove_repeated_content() -> None:
    bs_transformer = BeautifulSoupTransformer()
    with_lines_html = "<p>1\n1\n1\n1</p>"
    documents = [Document(page_content=with_lines_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == "1 1 1 1"


@pytest.mark.requires("bs4")
def test_extract_nested_tags() -> None:
    bs_transformer = BeautifulSoupTransformer()
    nested_html = (
        "<html><div class='some_style'>"
        "<p><span>First</span> paragraph.</p>"
        "<p>Second <div>paragraph.</div></p>"
        "<p><p>Third paragraph.</p></p>"
        "</div></html>"
    )
    documents = [Document(page_content=nested_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert (
        docs_transformed[0].page_content
        == "First paragraph. Second paragraph. Third paragraph."
    )


@pytest.mark.requires("bs4")
def test_extract_more_nested_tags() -> None:
    bs_transformer = BeautifulSoupTransformer()
    nested_html = (
        "<html><div class='some_style'>"
        "<p><span>First</span> paragraph.</p>"
        "<p>Second paragraph.</p>"
        "<p>Third paragraph with a list:"
        "<ul>"
        "<li>First list item.</li>"
        "<li>Second list item.</li>"
        "</ul>"
        "</p>"
        "<p>Fourth paragraph.</p>"
        "</div></html>"
    )
    documents = [Document(page_content=nested_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "First paragraph. Second paragraph. "
        "Third paragraph with a list: "
        "First list item. Second list item. "
        "Fourth paragraph."
    )


@pytest.mark.requires("bs4")
def test_transform_keeps_order() -> None:
    bs_transformer = BeautifulSoupTransformer()
    multiple_tags_html = (
        "<h1>First heading.</h1>"
        "<p>First paragraph.</p>"
        "<h1>Second heading.</h1>"
        "<p>Second paragraph.</p>"
    )
    documents = [Document(page_content=multiple_tags_html)]

    # Order of "p" and "h1" in the "tags_to_extract" parameter is NOT important here:
    # it will keep the order of the original HTML.
    docs_transformed_p_then_h1 = bs_transformer.transform_documents(
        documents, tags_to_extract=["p", "h1"]
    )
    assert (
        docs_transformed_p_then_h1[0].page_content
        == "First heading. First paragraph. Second heading. Second paragraph."
    )

    # Recreating `documents` because transform_documents() modifies it.
    documents = [Document(page_content=multiple_tags_html)]

    # changing the order of "h1" and "p" in "tags_to_extract" does NOT flip the order
    # of the extracted tags:
    docs_transformed_h1_then_p = bs_transformer.transform_documents(
        documents, tags_to_extract=["h1", "p"]
    )
    assert (
        docs_transformed_h1_then_p[0].page_content
        == "First heading. First paragraph. Second heading. Second paragraph."
    )


@pytest.mark.requires("bs4")
def test_extracts_href() -> None:
    bs_transformer = BeautifulSoupTransformer()
    multiple_tags_html = (
        "<h1>First heading.</h1>"
        "<p>First paragraph with an <a href='http://example.com'>example</a></p>"
        "<p>Second paragraph with an <a>a tag without href</a></p>"
    )
    documents = [Document(page_content=multiple_tags_html)]

    docs_transformed = bs_transformer.transform_documents(
        documents, tags_to_extract=["p"]
    )
    assert docs_transformed[0].page_content == (
        "First paragraph with an example (http://example.com) "
        "Second paragraph with an a tag without href"
    )


@pytest.mark.requires("bs4")
def test_invalid_html() -> None:
    bs_transformer = BeautifulSoupTransformer()
    invalid_html_1 = "<html><h1>First heading."
    invalid_html_2 = "<html 1234 xyz"
    documents = [
        Document(page_content=invalid_html_1),
        Document(page_content=invalid_html_2),
    ]

    docs_transformed = bs_transformer.transform_documents(
        documents, tags_to_extract=["h1"]
    )
    assert docs_transformed[0].page_content == "First heading."
    assert docs_transformed[1].page_content == ""


@pytest.mark.requires("bs4")
def test_remove_comments() -> None:
    bs_transformer = BeautifulSoupTransformer()
    html_with_comments = (
        "<html><!-- Google tag (gtag.js) --><p>First paragraph.</p</html>"
    )
    documents = [
        Document(page_content=html_with_comments),
    ]

    docs_transformed = bs_transformer.transform_documents(
        documents, tags_to_extract=["html"], remove_comments=True
    )
    assert docs_transformed[0].page_content == "First paragraph."


@pytest.mark.requires("bs4")
def test_do_not_remove_comments() -> None:
    bs_transformer = BeautifulSoupTransformer()
    html_with_comments = (
        "<html><!-- Google tag (gtag.js) --><p>First paragraph.</p</html>"
    )
    documents = [
        Document(page_content=html_with_comments),
    ]

    docs_transformed = bs_transformer.transform_documents(
        documents,
        tags_to_extract=["html"],
    )
    assert docs_transformed[0].page_content == "Google tag (gtag.js) First paragraph."

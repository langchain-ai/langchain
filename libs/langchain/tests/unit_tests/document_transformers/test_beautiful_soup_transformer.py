"""Unit tests for beautiful soup document transformer."""
import pytest

from langchain.document_transformers import BeautifulSoupTransformer
from langchain.schema.document import Document


@pytest.mark.requires("bs4")
def test_transform_empty_html() -> None:
    bs_transformer = BeautifulSoupTransformer()
    empty_html = "<html></html>"
    documents = [Document(page_content=empty_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == ""


@pytest.mark.requires("bs4")
def test_extract_paragraph() -> None:
    bs_transformer = BeautifulSoupTransformer()
    paragraphs_html = "<html><p>First paragraph.</p><p>Second paragraph.</p><html>"
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == "First paragraph. Second paragraph."


@pytest.mark.requires("bs4")
def test_remove_style() -> None:
    bs_transformer = BeautifulSoupTransformer()
    with_style_html = (
        "<html><style>my_funky_style</style><p>First paragraph.</p></html>"
    )
    documents = [Document(page_content=with_style_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == "First paragraph."


@pytest.mark.requires("bs4")
def test_remove_unwanted_lines() -> None:
    bs_transformer = BeautifulSoupTransformer()
    with_lines_html = "<html>\n\n<p>First \n\n     paragraph.</p>\n</html>\n\n"
    documents = [Document(page_content=with_lines_html)]
    docs_transformed = bs_transformer.transform_documents(documents)
    assert docs_transformed[0].page_content == "First paragraph."


# FIXME: This test proves that the order of the tags is NOT preserved.
#        Documenting the current behavior here, but this should be fixed.
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

    # order of "p" and "h1" in the "tags_to_extract" parameter is important here:
    # it will first extract all "p" tags, then all "h1" tags, breaking the order
    # of the HTML.
    docs_transformed_p_then_h1 = bs_transformer.transform_documents(
        documents, tags_to_extract=["p", "h1"]
    )
    assert (
        docs_transformed_p_then_h1[0].page_content
        == "First paragraph. Second paragraph. First heading. Second heading."
    )

    # Recreating `documents` because transform_documents() modifies it.
    documents = [Document(page_content=multiple_tags_html)]

    # changing the order of "h1" and "p" in "tags_to_extract" flips the order of
    # the extracted tags:
    docs_transformed_h1_then_p = bs_transformer.transform_documents(
        documents, tags_to_extract=["h1", "p"]
    )
    assert (
        docs_transformed_h1_then_p[0].page_content
        == "First heading. Second heading. First paragraph. Second paragraph."
    )

    # The correct result should be:
    #
    #   "First heading. First paragraph. Second heading. Second paragraph."
    #
    # That is the order in the original HTML, that should be preserved to preserve
    # the semantic "meaning" of the text.

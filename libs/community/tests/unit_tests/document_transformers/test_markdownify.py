"""Unit tests for markdownify document transformer."""

import pytest
from langchain_core.documents import Document

from langchain_community.document_transformers import MarkdownifyTransformer


@pytest.mark.requires("markdownify")
def test_empty_html() -> None:
    markdownify = MarkdownifyTransformer()
    empty_html = "<html></html>"
    documents = [Document(page_content=empty_html)]
    docs_transformed = markdownify.transform_documents(documents)
    assert docs_transformed[0].page_content == ""


@pytest.mark.requires("markdownify")
def test_extract_paragraphs() -> None:
    markdownify = MarkdownifyTransformer()
    paragraphs_html = (
        "<html><h1>Header</h1><p>First paragraph.</p>"
        "<p>Second paragraph.</p><h1>Ignore at end</h1></html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = markdownify.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# Header\n\n" "First paragraph.\n\n" "Second paragraph.\n\n" "# Ignore at end"
    )


@pytest.mark.requires("markdownify")
def test_extract_html() -> None:
    markdownify = MarkdownifyTransformer(skip="title")
    basic_html = (
        "<!DOCTYPE html>"
        '<html lang="en">'
        "<head>"
        '    <meta charset="UTF-8">'
        "    <title>Simple Test Page</title>"
        "</head>"
        "<body>"
        "    <h1>Test Header</h1>"
        "    <p>First paragraph.</p>"
        "    <p>Second paragraph.</p>"
        '    <a href="https://example.com">Example Link</a>'
        "</body>"
        "</html>"
    )
    documents = [Document(page_content=basic_html)]
    docs_transformed = markdownify.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "Simple Test Page "
        "# Test Header\n\n "
        "First paragraph.\n\n "
        "Second paragraph.\n\n "
        "[Example Link](https://example.com)"
    )


@pytest.mark.requires("markdownify")
def test_strip_tags() -> None:
    markdownify = MarkdownifyTransformer(strip="strong")
    paragraphs_html = (
        "<html>"
        "<h1>Header</h1>"
        "   <p><strong>1st paragraph.</strong></p>"
        '   <p>2nd paragraph. Here is <a href="http://example.com">link</a></p>'
        '   <img src="image.jpg" alt="Sample Image">'
        "<h1>Ignore at end</h1></html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = markdownify.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# Header\n\n "
        "1st paragraph.\n\n "
        "2nd paragraph. Here is [link](http://example.com)\n\n "
        "![Sample Image](image.jpg)"
        "# Ignore at end"
    )

    markdownify = MarkdownifyTransformer(strip=["strong", "a", "img"])
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = markdownify.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# Header\n\n "
        "1st paragraph.\n\n "
        "2nd paragraph. Here is link\n\n "
        "# Ignore at end"
    )


@pytest.mark.requires("markdownify")
def test_convert_tags() -> None:
    markdownify = MarkdownifyTransformer(convert=["strong", "a"])
    paragraphs_html = (
        "<html>"
        "<h1>Header</h1>"
        "   <p><strong>1st paragraph.</strong></p>"
        '   <p>2nd paragraph. Here is <a href="http://example.com">link</a></p>'
        '   <img src="image.jpg" alt="Sample Image">'
        "<h1>Ignore at end</h1></html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = markdownify.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "Header "
        "**1st paragraph.** "
        "2nd paragraph. "
        "Here is [link](http://example.com) "
        "Ignore at end"
    )

    markdownify = MarkdownifyTransformer(convert="p")
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = markdownify.transform_documents(documents)
    assert docs_transformed[0].page_content == (
        "Header "
        "1st paragraph.\n\n "
        "2nd paragraph. Here is link\n\n "
        "Ignore at end"
    )


@pytest.mark.requires("markdownify")
def test_strip_convert_conflict_error() -> None:
    with pytest.raises(
        ValueError,
        match="You may specify either tags to strip or tags to convert, but not both.",
    ):
        markdownify = MarkdownifyTransformer(strip="h1", convert=["strong", "a"])
        paragraphs_html = (
            "<html>"
            "<h1>Header</h1>"
            "   <p><strong>1st paragraph.</strong></p>"
            '   <p>2nd paragraph. Here is <a href="http://example.com">link</a></p>'
            '   <img src="image.jpg" alt="Sample Image">'
            "<h1>Ignore at end</h1></html>"
        )
        documents = [Document(page_content=paragraphs_html)]
        markdownify.transform_documents(documents)


# Async variants: exact duplicates of the above functions, using atransform_documents()
@pytest.mark.requires("markdownify")
async def test_empty_html_async() -> None:
    markdownify = MarkdownifyTransformer()
    empty_html = "<html></html>"
    documents = [Document(page_content=empty_html)]
    docs_transformed = await markdownify.atransform_documents(documents)
    assert docs_transformed[0].page_content == ""


@pytest.mark.requires("markdownify")
async def test_extract_paragraphs_async() -> None:
    markdownify = MarkdownifyTransformer()
    paragraphs_html = (
        "<html><h1>Header</h1><p>First paragraph.</p>"
        "<p>Second paragraph.</p><h1>Ignore at end</h1></html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = await markdownify.atransform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# Header\n\n" "First paragraph.\n\n" "Second paragraph.\n\n" "# Ignore at end"
    )


@pytest.mark.requires("markdownify")
async def test_extract_html_async() -> None:
    markdownify = MarkdownifyTransformer(skip="title")
    basic_html = (
        "<!DOCTYPE html>"
        '<html lang="en">'
        "<head>"
        '    <meta charset="UTF-8">'
        "    <title>Simple Test Page</title>"
        "</head>"
        "<body>"
        "    <h1>Test Header</h1>"
        "    <p>First paragraph.</p>"
        "    <p>Second paragraph.</p>"
        '    <a href="https://example.com">Example Link</a>'
        "</body>"
        "</html>"
    )
    documents = [Document(page_content=basic_html)]
    docs_transformed = await markdownify.atransform_documents(documents)
    assert docs_transformed[0].page_content == (
        "Simple Test Page "
        "# Test Header\n\n "
        "First paragraph.\n\n "
        "Second paragraph.\n\n "
        "[Example Link](https://example.com)"
    )


@pytest.mark.requires("markdownify")
async def test_strip_tags_async() -> None:
    markdownify = MarkdownifyTransformer(strip="strong")
    paragraphs_html = (
        "<html>"
        "<h1>Header</h1>"
        "   <p><strong>1st paragraph.</strong></p>"
        '   <p>2nd paragraph. Here is <a href="http://example.com">link</a></p>'
        '   <img src="image.jpg" alt="Sample Image">'
        "<h1>Ignore at end</h1></html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = await markdownify.atransform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# Header\n\n "
        "1st paragraph.\n\n "
        "2nd paragraph. Here is [link](http://example.com)\n\n "
        "![Sample Image](image.jpg)"
        "# Ignore at end"
    )

    markdownify = MarkdownifyTransformer(strip=["strong", "a", "img"])
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = await markdownify.atransform_documents(documents)
    assert docs_transformed[0].page_content == (
        "# Header\n\n "
        "1st paragraph.\n\n "
        "2nd paragraph. Here is link\n\n "
        "# Ignore at end"
    )


@pytest.mark.requires("markdownify")
async def test_convert_tags_async() -> None:
    markdownify = MarkdownifyTransformer(convert=["strong", "a"])
    paragraphs_html = (
        "<html>"
        "<h1>Header</h1>"
        "   <p><strong>1st paragraph.</strong></p>"
        '   <p>2nd paragraph. Here is <a href="http://example.com">link</a></p>'
        '   <img src="image.jpg" alt="Sample Image">'
        "<h1>Ignore at end</h1></html>"
    )
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = await markdownify.atransform_documents(documents)
    assert docs_transformed[0].page_content == (
        "Header "
        "**1st paragraph.** "
        "2nd paragraph. "
        "Here is [link](http://example.com) "
        "Ignore at end"
    )

    markdownify = MarkdownifyTransformer(convert="p")
    documents = [Document(page_content=paragraphs_html)]
    docs_transformed = await markdownify.atransform_documents(documents)
    assert docs_transformed[0].page_content == (
        "Header "
        "1st paragraph.\n\n "
        "2nd paragraph. Here is link\n\n "
        "Ignore at end"
    )


@pytest.mark.requires("markdownify")
async def test_strip_convert_conflict_error_async() -> None:
    with pytest.raises(
        ValueError,
        match="You may specify either tags to strip or tags to convert, but not both.",
    ):
        markdownify = MarkdownifyTransformer(strip="h1", convert=["strong", "a"])
        paragraphs_html = (
            "<html>"
            "<h1>Header</h1>"
            "   <p><strong>1st paragraph.</strong></p>"
            '   <p>2nd paragraph. Here is <a href="http://example.com">link</a></p>'
            '   <img src="image.jpg" alt="Sample Image">'
            "<h1>Ignore at end</h1></html>"
        )
        documents = [Document(page_content=paragraphs_html)]
        await markdownify.atransform_documents(documents)

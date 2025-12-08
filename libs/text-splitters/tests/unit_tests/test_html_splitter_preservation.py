import pytest
from langchain_text_splitters import HTMLSemanticPreservingSplitter

def test_preserve_nested_elements() -> None:
    body = """
    <p>text before <span><preserved>content</preserved></span> text after</p>
    """
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[],
        elements_to_preserve=["preserved"],
    )
    docs = splitter.split_text(body)
    assert len(docs) == 1
    # Check that <preserved> tag is intact
    assert "<preserved>content</preserved>" in docs[0].page_content

def test_preserve_container_elements() -> None:
    body = """
    <div>
        <preserved-container>
            <p>Content inside</p>
        </preserved-container>
    </div>
    """
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[],
        elements_to_preserve=["preserved-container"],
    )
    docs = splitter.split_text(body)
    assert len(docs) == 1
    # The container itself should be preserved, not unwrapped
    assert "<preserved-container>" in docs[0].page_content
    assert "</preserved-container>" in docs[0].page_content

def test_preserve_body_element() -> None:
    # This was the original issue report case
    body = """
    <p>Hello1<body>nest\n\n\n\nedbody</body></p>
    """
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[],
        elements_to_preserve=["body"],
    )
    docs = splitter.split_text(body)
    assert len(docs) == 1
    assert "<body>nest" in docs[0].page_content
    assert "edbody</body>" in docs[0].page_content

def test_nested_preservation_with_multiple_elements() -> None:
    body = """
    <outer>
        <inner>preserved text</inner>
    </outer>
    """
    # If we preserve outer, inner should remain part of outer's text string
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[],
        elements_to_preserve=["outer"],
    )
    docs = splitter.split_text(body)
    assert "<outer>" in docs[0].page_content
    assert "<inner>preserved text</inner>" in docs[0].page_content


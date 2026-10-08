"""Tests for document context formatting."""

from langchain_core.documents import Document, format_context


def test_format_context_empty() -> None:
    assert format_context([]) == ""


def test_format_context_zero_indexed_page() -> None:
    docs = [
        Document(
            page_content="Sample text",
            metadata={"source": "report.pdf", "page": 0},
        )
    ]
    expected = "[Source: report.pdf | Page: 1]\nSample text"
    assert format_context(docs) == expected


def test_format_context_one_indexed_page() -> None:
    docs = [
        Document(
            page_content="Sample text",
            metadata={"source": "report.pdf", "page": 1},
        )
    ]
    expected = "[Source: report.pdf | Page: 2]\nSample text"
    assert format_context(docs) == expected


def test_format_context_page_label() -> None:
    docs = [
        Document(
            page_content="Intro text",
            metadata={"source": "book.pdf", "page_label": "iv"},
        )
    ]
    expected = "[Source: book.pdf | Page: iv]\nIntro text"
    assert format_context(docs) == expected


def test_format_context_unpaged_document() -> None:
    docs = [
        Document(
            page_content="a,b\n1,2",
            metadata={"source": "data.csv"},
        )
    ]
    expected = "[Source: data.csv]\na,b\n1,2"
    assert format_context(docs) == expected


def test_format_context_file_name_precedence() -> None:
    docs = [
        Document(
            page_content="Content",
            metadata={
                "file_name": "actual.pdf",
                "source": "/tmp/upload_123/actual.pdf",
                "page": 1,
            },
        )
    ]
    expected = "[Source: actual.pdf | Page: 2]\nContent"
    assert format_context(docs) == expected


def test_format_context_missing_metadata() -> None:
    docs = [Document(page_content="No metadata here")]
    expected = "[Source: Document]\nNo metadata here"
    assert format_context(docs) == expected


def test_format_context_multiple_documents_and_custom_separator() -> None:
    docs = [
        Document(
            page_content="Page 1",
            metadata={"source": "doc.pdf", "page": 0},
        ),
        Document(
            page_content="CSV row",
            metadata={"source": "table.csv"},
        ),
    ]
    expected = (
        "[Source: doc.pdf | Page: 1]\nPage 1\n\n===\n\n[Source: table.csv]\nCSV row"
    )
    assert format_context(docs, separator="\n\n===\n\n") == expected

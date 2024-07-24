import os
from pathlib import Path

from langchain_community.document_loaders import (
    DedocAPIFileLoader,
    DedocFileLoader,
    DedocPDFLoader,
)

EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent / "examples/")

FILE_NAMES = [
    "example.html",
    "example.json",
    "fake-email-attachment.eml",
    "layout-parser-paper.pdf",
    "slack_export.zip",
    "stanley-cups.csv",
    "stanley-cups.xlsx",
    "whatsapp_chat.txt",
]


def test_dedoc_file_loader() -> None:
    for file_name in FILE_NAMES:
        file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, file_name)
        loader = DedocFileLoader(
            file_path,
            split="document",
            with_tables=False,
            pdf_with_text_layer="tabby",
            pages=":1",
        )
        docs = loader.load()

        assert len(docs) == 1


def test_dedoc_pdf_loader() -> None:
    file_name = "layout-parser-paper.pdf"
    for mode in ("true", "tabby"):
        file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, file_name)
        loader = DedocPDFLoader(
            file_path,
            split="document",
            with_tables=False,
            pdf_with_text_layer=mode,
            pages=":1",
        )
        docs = loader.load()

        assert len(docs) == 1


def test_dedoc_content_html() -> None:
    file_name = "example.html"
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, file_name)
    loader = DedocFileLoader(
        file_path,
        split="line",
        with_tables=False,
    )
    docs = loader.load()

    assert docs[0].metadata["file_name"] == "example.html"
    assert docs[0].metadata["file_type"] == "text/html"
    assert "Instead of drinking water from the cat bowl" in docs[0].page_content
    assert "Chase the red dot" not in docs[0].page_content


def test_dedoc_content_pdf() -> None:
    file_name = "layout-parser-paper.pdf"
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, file_name)
    loader = DedocFileLoader(
        file_path, split="page", pdf_with_text_layer="tabby", pages=":5"
    )
    docs = loader.load()
    table_list = [item for item in docs if item.metadata.get("type", "") == "table"]

    assert len(docs) == 6
    assert docs[0].metadata["file_name"] == "layout-parser-paper.pdf"
    assert docs[0].metadata["file_type"] == "application/pdf"
    assert "This paper introduces LayoutParser, an open-source" in docs[0].page_content
    assert "layout detection [38, 22], table detection [26]" in docs[1].page_content
    assert "LayoutParser: A Uniﬁed Toolkit for DL-Based DIA" in docs[2].page_content
    assert len(table_list) > 0
    assert (
        '\n<tbody>\n<tr>\n<td colspan="1" rowspan="1">'
        in table_list[0].metadata["text_as_html"]
    )


def test_dedoc_content_json() -> None:
    file_name = "example.json"
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, file_name)
    loader = DedocFileLoader(file_path, split="node")
    docs = loader.load()

    assert len(docs) == 11
    assert docs[0].metadata["file_name"] == "example.json"
    assert docs[0].metadata["file_type"] == "application/json"
    assert "Bye!" in docs[0].page_content


def test_dedoc_content_txt() -> None:
    file_name = "whatsapp_chat.txt"
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, file_name)
    loader = DedocFileLoader(file_path, split="line")
    docs = loader.load()

    assert len(docs) == 10
    assert docs[0].metadata["file_name"] == "whatsapp_chat.txt"
    assert docs[0].metadata["file_type"] == "text/plain"
    assert "[05.05.23, 15:48:11] James: Hi here" in docs[0].page_content
    assert "[11/8/21, 9:41:32 AM] User name: Message 123" in docs[1].page_content
    assert "1/23/23, 3:19 AM - User 2: Bye!" in docs[2].page_content


def test_dedoc_table_handling() -> None:
    file_name = "stanley-cups.csv"
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, file_name)
    loader = DedocFileLoader(file_path, split="document")
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].metadata["file_name"] == "stanley-cups.csv"
    assert docs[0].metadata["file_type"] == "text/csv"
    assert docs[1].metadata["type"] == "table"
    assert '<td colspan="1" rowspan="1">1</td>' in docs[1].metadata["text_as_html"]
    assert "Maple Leafs\tTOR\t13" in docs[1].page_content


def test_dedoc_api_file_loader() -> None:
    file_name = "whatsapp_chat.txt"
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, file_name)
    loader = DedocAPIFileLoader(
        file_path, split="line", url="https://dedoc-readme.hf.space"
    )
    docs = loader.load()

    assert len(docs) == 10
    assert docs[0].metadata["file_name"] == "whatsapp_chat.txt"
    assert docs[0].metadata["file_type"] == "text/plain"
    assert "[05.05.23, 15:48:11] James: Hi here" in docs[0].page_content
    assert "[11/8/21, 9:41:32 AM] User name: Message 123" in docs[1].page_content
    assert "1/23/23, 3:19 AM - User 2: Bye!" in docs[2].page_content

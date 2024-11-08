from pathlib import Path

import pytest

from langchain_community.document_loaders import DoclingLoader

HELLO_PDF = Path(__file__).parent.parent.parent / "examples" / "hello.pdf"


@pytest.mark.requires("docling")
def test_docling_load_as_markdown() -> None:
    loader = DoclingLoader(
        file_path=str(HELLO_PDF.absolute()),
        export_type=DoclingLoader.ExportType.MARKDOWN,
    )
    docs = loader.load()
    assert len(docs) == 1
    assert "Hello world!" in docs[0].page_content

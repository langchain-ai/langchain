from pathlib import Path

import pytest

from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.parsers.documentloader_adapter import (
    DocumentLoaderAsParser,
)

HERE = Path(__file__).parent
EXAMPLES = HERE.parent.parent.parent / "examples"


@pytest.mark.requires("openpyxl")
def test_excel_loader_as_parser() -> None:
    """Test DocumentLoaderAsParser with UnstructuredExcelLoader."""
    file_path = EXAMPLES / "stanley-cups.xlsx"
    blob = Blob.from_path(file_path)

    # Initialize the parser adapter with UnstructuredExcelLoader
    parser = DocumentLoaderAsParser(UnstructuredExcelLoader, mode="single")

    # Parse the blob and retrieve documents
    docs = list(parser.lazy_parse(blob))

    # Verify that the result is a list and check content and metadata
    assert isinstance(docs, list)
    assert len(docs) > 0  # Check that at least one document was parsed

    # Extract metadata and content from the first document
    metadata = docs[0].metadata
    content = docs[0].page_content

    # Assert metadata and content are as expected
    assert metadata["source"] == file_path
    assert "Stanley Cups" in content
    assert "Maple Leafs" in content  # Check for expected data in the content

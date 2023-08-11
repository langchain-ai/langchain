import os
from pathlib import Path

from langchain.document_loaders import UnstructuredExcelLoader

EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "examples"


def test_unstructured_excel_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DIRECTORY, "stanley-cups.xlsx")
    loader = UnstructuredExcelLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

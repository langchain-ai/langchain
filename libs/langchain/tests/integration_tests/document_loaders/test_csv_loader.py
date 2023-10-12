import os
from pathlib import Path

from langchain.document_loaders import UnstructuredCSVLoader

EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "examples"


def test_unstructured_csv_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DIRECTORY, "stanley-cups.csv")
    loader = UnstructuredCSVLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

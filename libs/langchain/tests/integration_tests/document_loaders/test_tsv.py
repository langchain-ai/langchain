import os
from pathlib import Path

from langchain.document_loaders import UnstructuredTSVLoader

EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "examples"


def test_unstructured_tsv_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DIRECTORY, "stanley-cups.tsv")
    loader = UnstructuredTSVLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

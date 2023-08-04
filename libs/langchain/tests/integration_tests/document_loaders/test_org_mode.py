import os
from pathlib import Path

from langchain.document_loaders import UnstructuredOrgModeLoader

EXAMPLE_DIRECTORY = file_path = Path(__file__).parent.parent / "examples"


def test_unstructured_org_mode_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DIRECTORY, "README.org")
    loader = UnstructuredOrgModeLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

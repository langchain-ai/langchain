from pathlib import Path

from langchain.document_loaders import UnstructuredODTLoader


def test_unstructured_odt_loader() -> None:
    """Test unstructured loader."""
    file_path = Path(__file__).parent.parent / "examples/fake.odt"
    loader = UnstructuredODTLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

from pathlib import Path

import pytest

from langchain.document_loaders.python import PythonLoader


@pytest.mark.parametrize("filename", ["default-encoding.py", "non-utf8-encoding.py"])
def test_python_loader(filename: str) -> None:
    """Test Python loader."""
    file_path = Path(__file__).parent.parent / "examples" / filename
    loader = PythonLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

    metadata = docs[0].metadata

    assert metadata["source"] == str(file_path)

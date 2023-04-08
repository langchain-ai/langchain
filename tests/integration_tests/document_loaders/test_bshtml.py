import sys
from pathlib import Path

import pytest

from langchain.document_loaders.html_bs import BSHTMLLoader


def test_bs_html_loader() -> None:
    """Test unstructured loader."""
    file_path = Path(__file__).parent.parent / "examples/example.html"
    loader = BSHTMLLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

    metadata = docs[0].metadata

    assert metadata["title"] == "Chew dad's slippers"
    assert metadata["source"] == str(file_path)


@pytest.mark.skipif(
    bool(sys.flags.utf8_mode) or not sys.platform.startswith("win"),
    reason="default encoding is utf8",
)
def test_bs_html_loader_non_utf8() -> None:
    """Test providing encoding to BSHTMLLoader."""
    file_path = Path(__file__).parent.parent / "examples/example-utf8.html"

    with pytest.raises(UnicodeDecodeError):
        BSHTMLLoader(str(file_path)).load()

    loader = BSHTMLLoader(str(file_path), open_encoding="utf8")
    docs = loader.load()

    assert len(docs) == 1

    metadata = docs[0].metadata

    assert metadata["title"] == "Chew dad's slippers"
    assert metadata["source"] == str(file_path)

import sys
from pathlib import Path

import pytest

from langchain.document_loaders.html_bs import BSHTMLLoader

HERE = Path(__file__).parent
EXAMPLES = HERE.parent.parent / "integration_tests" / "examples"


@pytest.mark.requires("bs4", "lxml")
def test_bs_html_loader() -> None:
    """Test unstructured loader."""
    file_path = EXAMPLES / "example.html"
    loader = BSHTMLLoader(str(file_path), get_text_separator="|")
    docs = loader.load()

    assert len(docs) == 1

    metadata = docs[0].metadata
    content = docs[0].page_content

    assert metadata["title"] == "Chew dad's slippers"
    assert metadata["source"] == str(file_path)
    assert content[:2] == "\n|"


@pytest.mark.skipif(
    bool(sys.flags.utf8_mode) or not sys.platform.startswith("win"),
    reason="default encoding is utf8",
)
@pytest.mark.requires("bs4", "lxml")
def test_bs_html_loader_non_utf8() -> None:
    """Test providing encoding to BSHTMLLoader."""
    file_path = EXAMPLES / "example-utf8.html"

    with pytest.raises(UnicodeDecodeError):
        BSHTMLLoader(str(file_path)).load()

    loader = BSHTMLLoader(str(file_path), open_encoding="utf8")
    docs = loader.load()

    assert len(docs) == 1

    metadata = docs[0].metadata

    assert metadata["title"] == "Chew dad's slippers"
    assert metadata["source"] == str(file_path)

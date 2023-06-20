from pathlib import Path

import pytest

from langchain.document_loaders.mhtml import MHTMLLoader

HERE = Path(__file__).parent
EXAMPLES = HERE.parent.parent / "integration_tests" / "examples"


@pytest.mark.requires("bs4", "lxml")
def test_mhtml_loader() -> None:
    """Test mhtml loader."""
    file_path = EXAMPLES / "example.mht"
    loader = MHTMLLoader(str(file_path))
    docs = loader.load()

    assert len(docs) == 1

    metadata = docs[0].metadata
    content = docs[0].page_content

    assert metadata["title"] == "LangChain"
    assert metadata["source"] == str(file_path)
    assert "LANG CHAIN 🦜️🔗Official Home Page" in content

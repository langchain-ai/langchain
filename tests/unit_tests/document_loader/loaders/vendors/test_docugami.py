"""Test DocugamiLoader."""
from pathlib import Path

import pytest

from langchain.document_loaders import DocugamiLoader

DOCUGAMI_XML_PATH = Path(__file__).parent / "test_data" / "docugami-example.xml"


@pytest.mark.requires("lxml")
def test_docugami_loader_local() -> None:
    """Test DocugamiLoader."""
    loader = DocugamiLoader(file_paths=[DOCUGAMI_XML_PATH])
    docs = loader.load()

    assert len(docs) == 19

    xpath = docs[0].metadata.get("xpath")
    assert str(xpath).endswith("/docset:Preamble")
    assert docs[0].metadata["structure"] == "p"
    assert docs[0].metadata["tag"] == "Preamble"
    assert docs[0].page_content.startswith("MUTUAL NON-DISCLOSURE AGREEMENT")


def test_docugami_initialization() -> None:
    """Test correct initialization in remote mode."""
    DocugamiLoader(access_token="test", docset_id="123")

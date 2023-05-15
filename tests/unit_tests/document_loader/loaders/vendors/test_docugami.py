"""Test DocugamiLoader."""
import pytest
from pathlib import Path

from langchain.document_loaders import DocugamiLoader


@pytest.mark.requires("docugami")
def test_docugami_loader_local() -> None:
    """Test DocugamiLoader."""
    file_path = Path(__file__).parent / "../examples/docugami-example.xml"
    loader = DocugamiLoader(file_paths=[file_path])
    docs = loader.load()

    assert len(docs) == 19

    xpath = docs[0].metadata.get("xpath")
    assert str(xpath).endswith("/docset:Preamble")
    assert docs[0].metadata["structure"] == "p"
    assert docs[0].metadata["tag"] == "Preamble"
    assert docs[0].page_content.startswith("MUTUAL NON-DISCLOSURE AGREEMENT")


def test_docugami_loader_remote_init() -> None:
    """Test correct initialization in remote mode."""
    DocugamiLoader(access_token="test", docset_id="123")

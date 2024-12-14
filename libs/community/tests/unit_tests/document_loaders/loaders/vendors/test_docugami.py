"""Test DocugamiLoader."""

from pathlib import Path

import pytest

from langchain_community.document_loaders import DocugamiLoader

DOCUGAMI_XML_PATH = Path(__file__).parent / "test_data" / "docugami-example.xml"


@pytest.mark.requires("dgml_utils")
def test_docugami_loader_local() -> None:
    """Test DocugamiLoader."""
    loader = DocugamiLoader(file_paths=[DOCUGAMI_XML_PATH])  # type: ignore[call-arg]
    docs = loader.load()

    assert len(docs) == 25

    assert "/docset:DisclosingParty" in docs[1].metadata["xpath"]
    assert "h1" in docs[1].metadata["structure"]
    assert "DisclosingParty" in docs[1].metadata["tag"]
    assert docs[1].page_content.startswith("Disclosing")


def test_docugami_initialization() -> None:
    """Test correct initialization in remote mode."""
    DocugamiLoader(
        access_token="test", docset_id="123", document_ids=None, file_paths=None
    )

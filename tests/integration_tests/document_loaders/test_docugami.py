from pathlib import Path

from langchain.document_loaders import DocugamiLoader


def test_docugami_loader_local() -> None:
    """Test DocugamiLoader."""
    file_path = Path(__file__).parent / "../examples/docugami-example.xml"
    loader = DocugamiLoader(file_paths=[file_path])
    docs = loader.load()

    assert len(docs) == 45

    xpath = docs[0].metadata.get("xpath")
    assert str(xpath).endswith("/docset:MutualNon-disclosure")
    assert docs[0].metadata["structure"] == "h1"
    assert docs[0].metadata["tag"] == "MutualNon-disclosure"
    assert docs[0].page_content == "MUTUAL NON-DISCLOSURE AGREEMENT"


def test_docugami_loader_remote_init() -> None:
    """Test correct initialization in remote mode."""
    _ = DocugamiLoader(access_token="test", docset_id="123")

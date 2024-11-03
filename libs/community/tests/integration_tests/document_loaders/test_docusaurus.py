from pathlib import Path

from langchain_community.document_loaders import DocusaurusLoader

DOCS_URL = str(Path(__file__).parent.parent / "examples/docusaurus-sitemap.xml")


def test_docusarus() -> None:
    """Test sitemap loader."""
    loader = DocusaurusLoader(DOCS_URL, is_local=True)
    documents = loader.load()
    assert len(documents) > 1
    assert "🦜️🔗 Langchain" in documents[0].page_content


def test_filter_docusaurus_sitemap() -> None:
    """Test sitemap loader."""
    loader = DocusaurusLoader(
        DOCS_URL,
        is_local=True,
        filter_urls=[
            "https://python.langchain.com/docs/integrations/document_loaders/sitemap"
        ],
    )
    documents = loader.load()
    assert len(documents) == 1
    assert "SitemapLoader" in documents[0].page_content


def test_docusarus_metadata() -> None:
    def sitemap_metadata_one(meta: dict, _content: None) -> dict:
        return {**meta, "mykey": "Super Important Metadata"}

    """Test sitemap loader."""
    loader = DocusaurusLoader(
        DOCS_URL,
        is_local=True,
        meta_function=sitemap_metadata_one,
    )
    documents = loader.load()
    assert len(documents) > 1
    assert "mykey" in documents[0].metadata
    assert "Super Important Metadata" in documents[0].metadata["mykey"]

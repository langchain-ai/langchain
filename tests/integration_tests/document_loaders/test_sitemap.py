from langchain.document_loaders import SitemapLoader


def test_sitemap() -> None:
    """Test sitemap loader."""
    loader = SitemapLoader("https://langchain.readthedocs.io/sitemap.xml")
    documents = loader.load()
    assert len(documents) > 1
    assert "🦜🔗" in documents[0].page_content


def test_filter_sitemap() -> None:
    """Test sitemap loader."""
    loader = SitemapLoader(
        "https://langchain.readthedocs.io/sitemap.xml",
        filter_urls=["https://langchain.readthedocs.io/en/stable/"],
    )
    documents = loader.load()
    assert len(documents) == 1
    assert "🦜🔗" in documents[0].page_content

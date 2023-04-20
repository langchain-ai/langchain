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


def test_discover_sitemap_url() -> None:
    """Test sitemap discovery from homepage."""
    loader = SitemapLoader("https://langchain.readthedocs.io/", discover_sitemap=True)
    documents = loader.load()
    assert len(documents) > 1
    assert "🦜🔗" in documents[0].page_content
    assert loader.web_path == "https://python.langchain.com/sitemap.xml"

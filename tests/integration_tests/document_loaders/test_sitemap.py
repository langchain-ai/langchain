import pytest

from langchain.document_loaders import SitemapLoader


def test_sitemap() -> None:
    """Test sitemap loader."""
    loader = SitemapLoader("https://langchain.readthedocs.io/sitemap.xml")
    documents = loader.load()
    assert len(documents) > 1
    assert "🦜🔗" in documents[0].page_content


def test_sitemap_block() -> None:
    """Test sitemap loader."""
    loader = SitemapLoader(
        "https://langchain.readthedocs.io/sitemap.xml", blocksize=1, blocknum=1
    )
    documents = loader.load()
    assert len(documents) == 1
    assert "🦜🔗" in documents[0].page_content


def test_sitemap_block_only_one() -> None:
    """Test sitemap loader."""
    loader = SitemapLoader(
        "https://langchain.readthedocs.io/sitemap.xml", blocksize=1000000, blocknum=0
    )
    documents = loader.load()
    assert len(documents) > 1
    assert "🦜🔗" in documents[0].page_content


def test_sitemap_block_blocknum_default() -> None:
    """Test sitemap loader."""
    loader = SitemapLoader(
        "https://langchain.readthedocs.io/sitemap.xml", blocksize=1000000
    )
    documents = loader.load()
    assert len(documents) > 1
    assert "🦜🔗" in documents[0].page_content


def test_sitemap_block_size_to_small() -> None:
    """Test sitemap loader."""
    with pytest.raises(ValueError, match="Sitemap blocksize should be at least 1"):
        SitemapLoader("https://langchain.readthedocs.io/sitemap.xml", blocksize=0)


def test_sitemap_block_num_to_small() -> None:
    """Test sitemap loader."""
    with pytest.raises(ValueError, match="Sitemap blocknum can not be lower then 0"):
        SitemapLoader(
            "https://langchain.readthedocs.io/sitemap.xml",
            blocksize=1000000,
            blocknum=-1,
        )


def test_sitemap_block_does_not_exists() -> None:
    """Test sitemap loader."""
    loader = SitemapLoader(
        "https://langchain.readthedocs.io/sitemap.xml", blocksize=1000000, blocknum=15
    )
    with pytest.raises(
        ValueError,
        match="Selected sitemap does not contain enough blocks for given blocknum",
    ):
        loader.load()


def test_filter_sitemap() -> None:
    """Test sitemap loader."""
    loader = SitemapLoader(
        "https://langchain.readthedocs.io/sitemap.xml",
        filter_urls=["https://python.langchain.com/en/stable/"],
    )
    documents = loader.load()
    assert len(documents) == 1
    assert "🦜🔗" in documents[0].page_content

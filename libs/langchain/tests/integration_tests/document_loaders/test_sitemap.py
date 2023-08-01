from pathlib import Path
from typing import Any

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


def test_sitemap_metadata() -> None:
    def sitemap_metadata_one(meta: dict, _content: None) -> dict:
        return {**meta, "mykey": "Super Important Metadata"}

    """Test sitemap loader."""
    loader = SitemapLoader(
        "https://langchain.readthedocs.io/sitemap.xml",
        meta_function=sitemap_metadata_one,
    )
    documents = loader.load()
    assert len(documents) > 1
    assert "mykey" in documents[0].metadata
    assert "Super Important Metadata" in documents[0].metadata["mykey"]


def test_sitemap_metadata_extraction() -> None:
    def sitemap_metadata_two(meta: dict, content: Any) -> dict:
        title = content.find("title")
        if title:
            return {**meta, "title": title.get_text()}
        return meta

    """Test sitemap loader."""
    loader = SitemapLoader(
        "https://langchain.readthedocs.io/sitemap.xml",
        meta_function=sitemap_metadata_two,
    )
    documents = loader.load()
    assert len(documents) > 1
    assert "title" in documents[0].metadata
    assert "LangChain" in documents[0].metadata["title"]


def test_sitemap_metadata_default() -> None:
    """Test sitemap loader."""
    loader = SitemapLoader("https://langchain.readthedocs.io/sitemap.xml")
    documents = loader.load()
    assert len(documents) > 1
    assert "source" in documents[0].metadata
    assert "loc" in documents[0].metadata


def test_local_sitemap() -> None:
    """Test sitemap loader."""
    file_path = Path(__file__).parent.parent / "examples/sitemap.xml"
    loader = SitemapLoader(str(file_path))
    documents = loader.load()
    assert len(documents) > 1
    assert "🦜🔗" in documents[0].page_content

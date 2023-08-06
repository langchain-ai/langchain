from pathlib import Path

from langchain.document_loaders.rss import RSSFeedLoader


def test_rss_loader() -> None:
    loader = RSSFeedLoader(urls=["https://www.engadget.com/rss.xml"])
    docs = loader.load()

    assert docs[0] is not None
    assert hasattr(docs[0], "page_content")
    assert hasattr(docs[0], "metadata")

    metadata = docs[0].metadata
    assert "feed" in metadata
    assert "title" in metadata
    assert "link" in metadata
    assert "authors" in metadata
    assert "language" in metadata
    assert "description" in metadata
    assert "publish_date" in metadata


def test_rss_loader_with_opml() -> None:
    file_path = Path(__file__).parent.parent / "examples"
    with open(file_path.joinpath("sample_rss_feeds.opml"), "r") as f:
        loader = RSSFeedLoader(opml=f.read())

    docs = loader.load()

    assert docs[0] is not None
    assert hasattr(docs[0], "page_content")
    assert hasattr(docs[0], "metadata")

    metadata = docs[0].metadata
    assert "feed" in metadata
    assert "title" in metadata
    assert "link" in metadata
    assert "authors" in metadata
    assert "language" in metadata
    assert "description" in metadata
    assert "publish_date" in metadata

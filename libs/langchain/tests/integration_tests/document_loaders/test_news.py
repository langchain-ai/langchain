import random

import pytest
import requests

from langchain.document_loaders import NewsURLLoader


def get_random_news_url() -> str:
    from bs4 import BeautifulSoup

    response = requests.get("https://news.google.com")
    soup = BeautifulSoup(response.text, "html.parser")

    article_links = [
        a["href"] for a in soup.find_all("a", href=True) if "/articles/" in a["href"]
    ]
    random_article_link = random.choice(article_links)

    return "https://news.google.com" + random_article_link


def test_news_loader() -> None:
    loader = NewsURLLoader([get_random_news_url()])
    docs = loader.load()

    assert docs[0] is not None
    assert hasattr(docs[0], "page_content")
    assert hasattr(docs[0], "metadata")

    metadata = docs[0].metadata
    assert "title" in metadata
    assert "link" in metadata
    assert "authors" in metadata
    assert "language" in metadata
    assert "description" in metadata
    assert "publish_date" in metadata


def test_news_loader_with_nlp() -> None:
    loader = NewsURLLoader([get_random_news_url()], nlp=True)
    docs = loader.load()

    assert docs[0] is not None
    assert hasattr(docs[0], "page_content")
    assert hasattr(docs[0], "metadata")

    metadata = docs[0].metadata
    assert "title" in metadata
    assert "link" in metadata
    assert "authors" in metadata
    assert "language" in metadata
    assert "description" in metadata
    assert "publish_date" in metadata
    assert "keywords" in metadata
    assert "summary" in metadata


def test_continue_on_failure_true() -> None:
    """Test exception is not raised when continue_on_failure=True."""
    loader = NewsURLLoader(["badurl.foobar"])
    loader.load()


def test_continue_on_failure_false() -> None:
    """Test exception is raised when continue_on_failure=False."""
    loader = NewsURLLoader(["badurl.foobar"], continue_on_failure=False)
    with pytest.raises(Exception):
        loader.load()

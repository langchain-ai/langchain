import pytest
import requests
from bs4 import BeautifulSoup as soup

from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader


@pytest.mark.requires("bs4", "requests", "aiohttp", "asyncio")
def test_async_recursive_url_loader() -> None:
    url = "https://js.langchain.com/docs/modules/memory/examples/"
    loader = RecursiveUrlLoader(
        url=url, extractor=lambda _: "placeholder", use_async=True
    )
    docs = loader.load()
    correct_doc_num = len(
        soup(requests.get(url).text, "html.parser").select("section.row > article")
    )
    assert len(docs) == correct_doc_num
    assert docs[0].page_content == "placeholder"


@pytest.mark.requires("bs4", "requests", "aiohttp", "asyncio")
def test_sync_recursive_url_loader() -> None:
    url = "https://js.langchain.com/docs/modules/memory/examples/"
    loader = RecursiveUrlLoader(
        url=url, extractor=lambda _: "placeholder", use_async=False
    )
    docs = loader.load()
    correct_doc_num = len(
        soup(requests.get(url).text, "html.parser").select("section.row > article")
    )
    assert len(docs) == correct_doc_num
    assert docs[0].page_content == "placeholder"


@pytest.mark.requires("bs4", "requests", "aiohttp", "asyncio")
def test_loading_invalid_url() -> None:
    url = "https://this.url.is.invalid/this/is/a/test"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda _: "placeholder", use_async=False
    )
    docs = loader.load()
    assert len(docs) == 0

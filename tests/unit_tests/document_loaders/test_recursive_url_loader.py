import requests
from bs4 import BeautifulSoup as soup

from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader

url = "https://js.langchain.com/docs/modules/memory/integrations/"


def get_correct_number() -> int:
    raw_html = requests.session().get(url).text
    return len(soup(raw_html, "html.parser").select("section.row > article"))


def test_async_recursive_url_loader() -> None:
    loader = RecursiveUrlLoader(
        url=url, extractor=lambda _: "placeholder", use_async=True
    )
    docs = loader.load()
    correct_doc_num = get_correct_number()
    assert len(docs) == correct_doc_num
    assert docs[0].page_content == "placeholder"


def test_sync_recursive_url_loader() -> None:
    loader = RecursiveUrlLoader(
        url=url, extractor=lambda _: "placeholder", use_async=False
    )
    docs = loader.load()
    correct_doc_num = get_correct_number()
    assert len(docs) == correct_doc_num
    assert docs[0].page_content == "placeholder"


def test_loading_invalid_url() -> None:
    url = "https://this.url.is.invalid/this/is/a/test"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda _: "placeholder", use_async=False
    )
    docs = loader.load()
    assert len(docs) == 0

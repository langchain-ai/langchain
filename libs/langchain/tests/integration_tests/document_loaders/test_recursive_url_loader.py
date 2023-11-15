import pytest as pytest

from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader


@pytest.mark.asyncio
def test_async_recursive_url_loader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url,
        extractor=lambda _: "placeholder",
        use_async=True,
        max_depth=3,
        timeout=None,
        check_response_status=True,
    )
    docs = loader.load()
    assert len(docs) == 513
    assert docs[0].page_content == "placeholder"


@pytest.mark.asyncio
def test_async_recursive_url_loader_deterministic() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url,
        use_async=True,
        max_depth=3,
        timeout=None,
    )
    docs = sorted(loader.load(), key=lambda d: d.metadata["source"])
    docs_2 = sorted(loader.load(), key=lambda d: d.metadata["source"])
    assert docs == docs_2


def test_sync_recursive_url_loader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url, extractor=lambda _: "placeholder", use_async=False, max_depth=2
    )
    docs = loader.load()
    assert len(docs) == 25
    assert docs[0].page_content == "placeholder"


@pytest.mark.asyncio
def test_sync_async_equivalent() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(url, use_async=False, max_depth=2)
    async_loader = RecursiveUrlLoader(url, use_async=False, max_depth=2)
    docs = sorted(loader.load(), key=lambda d: d.metadata["source"])
    async_docs = sorted(async_loader.load(), key=lambda d: d.metadata["source"])
    assert docs == async_docs


def test_loading_invalid_url() -> None:
    url = "https://this.url.is.invalid/this/is/a/test"
    loader = RecursiveUrlLoader(
        url, max_depth=1, extractor=lambda _: "placeholder", use_async=False
    )
    docs = loader.load()
    assert len(docs) == 0

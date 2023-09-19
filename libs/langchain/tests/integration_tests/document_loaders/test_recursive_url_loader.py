from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader


def test_async_recursive_url_loader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url=url, extractor=lambda _: "placeholder", use_async=True, max_depth=2
    )
    docs = loader.load()
    assert len(docs) == 27
    assert docs[0].page_content == "placeholder"


def test_sync_recursive_url_loader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url=url, extractor=lambda _: "placeholder", use_async=False, max_depth=2
    )
    docs = loader.load()
    assert len(docs) == 27
    assert docs[0].page_content == "placeholder"

    async_loader = RecursiveUrlLoader(
        url=url, extractor=lambda _: "placeholder", use_async=True, max_depth=2
    )
    async_docs = async_loader.load()
    assert sorted(docs, key=lambda d: d.page_content) == sorted(
        async_docs, key=lambda d: d.page_content
    )


def test_loading_invalid_url() -> None:
    url = "https://this.url.is.invalid/this/is/a/test"
    loader = RecursiveUrlLoader(
        url=url, max_depth=1, extractor=lambda _: "placeholder", use_async=False
    )
    docs = loader.load()
    assert len(docs) == 0

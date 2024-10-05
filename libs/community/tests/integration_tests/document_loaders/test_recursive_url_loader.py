import asyncio

from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader


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
    assert len(docs) == 512
    assert docs[0].page_content == "placeholder"


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


def test_async_recursive_url_lazy_loader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url,
        extractor=lambda _: "placeholder",
        use_async=True,
        max_depth=3,
        timeout=None,
        check_response_status=True,
    )
    docs = [doc for doc in loader.lazy_load()]
    assert len(docs) == 512
    assert docs[0].page_content == "placeholder"


def test_async_recursive_url_lazy_loader_deterministic() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url,
        use_async=True,
        max_depth=3,
        timeout=None,
    )
    docs = sorted(
        [doc for doc in loader.lazy_load()], key=lambda d: d.metadata["source"]
    )
    docs_2 = sorted(
        [doc for doc in loader.lazy_load()], key=lambda d: d.metadata["source"]
    )
    assert docs == docs_2


async def test_async_recursive_url_alazy_loader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url,
        extractor=lambda _: "placeholder",
        use_async=True,
        max_depth=3,
        timeout=None,
        check_response_status=True,
    )
    docs = [doc async for doc in loader.alazy_load()]
    assert len(docs) == 512
    assert docs[0].page_content == "placeholder"


def test_async_recursive_url_aloader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url,
        extractor=lambda _: "placeholder",
        use_async=True,
        max_depth=3,
        timeout=None,
        check_response_status=True,
    )
    docs = asyncio.run(loader.aload())
    assert len(docs) == 512
    assert docs[0].page_content == "placeholder"


def test_async_recursive_url_aloader_deterministic() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url,
        use_async=True,
        max_depth=3,
        timeout=None,
    )
    docs = sorted(asyncio.run(loader.aload()), key=lambda d: d.metadata["source"])
    docs_2 = sorted(asyncio.run(loader.aload()), key=lambda d: d.metadata["source"])
    assert docs == docs_2


async def test_async_recursive_url_alazy_loader_deterministic() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url,
        use_async=True,
        max_depth=3,
        timeout=None,
    )
    docs = sorted(
        [doc async for doc in loader.alazy_load()], key=lambda d: d.metadata["source"]
    )
    docs_2 = sorted(
        [doc async for doc in loader.alazy_load()], key=lambda d: d.metadata["source"]
    )
    assert docs == docs_2


def test_sync_recursive_url_loader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url, extractor=lambda _: "placeholder", use_async=False, max_depth=2
    )
    docs = loader.load()
    assert len(docs) == 24
    assert docs[0].page_content == "placeholder"


def test_sync_recursive_url_lazy_loader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url, extractor=lambda _: "placeholder", use_async=False, max_depth=2
    )
    docs = [doc for doc in loader.lazy_load()]
    assert len(docs) == 24
    assert docs[0].page_content == "placeholder"


def test_sync_recursive_url_aloader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url, extractor=lambda _: "placeholder", use_async=False, max_depth=2
    )
    docs = asyncio.run(loader.aload())
    assert len(docs) == 24
    assert docs[0].page_content == "placeholder"


async def test_sync_recursive_url_alazy_loader() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(
        url, extractor=lambda _: "placeholder", use_async=False, max_depth=2
    )
    docs = [doc async for doc in loader.alazy_load()]
    assert len(docs) == 24
    assert docs[0].page_content == "placeholder"


def test_sync_async_equivalent() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(url, use_async=False, max_depth=2)
    async_loader = RecursiveUrlLoader(url, use_async=True, max_depth=2)
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


def test_sync_async_metadata_necessary_properties() -> None:
    url = "https://docs.python.org/3.9/"
    loader = RecursiveUrlLoader(url, use_async=False, max_depth=2)
    async_loader = RecursiveUrlLoader(url, use_async=True, max_depth=2)
    docs = loader.load()
    async_docs = async_loader.load()
    for doc in docs:
        assert "source" in doc.metadata
        assert "content_type" in doc.metadata
    for doc in async_docs:
        assert "source" in doc.metadata
        assert "content_type" in doc.metadata

from langchain.document_loaders.async_recursive_url_loader import AsyncRecursiveUrlLoader

def test_async_recursive_url_loader() -> None:
    loader = AsyncRecursiveUrlLoader("https://langchain.com", raw_webpage_to_text_converter=lambda _: "x", max_depth=0)
    docs = loader.load()

    assert docs[0].page_content == "x"
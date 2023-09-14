from langchain.document_loaders.joplin import JoplinLoader


def test_joplin_loader() -> None:
    loader = JoplinLoader()
    docs = loader.load()

    assert isinstance(docs, list)
    assert isinstance(docs[0].page_content, str)
    assert isinstance(docs[0].metadata["source"], str)
    assert isinstance(docs[0].metadata["title"], str)

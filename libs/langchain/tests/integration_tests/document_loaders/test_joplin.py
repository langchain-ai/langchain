from langchain.document_loaders.joplin import JoplinLoader


def test_joplin_loader() -> None:
    loader = JoplinLoader()
    docs = loader.load()

    assert type(docs) is list
    assert type(docs[0].page_content) is str
    assert type(docs[0].metadata["source"]) is str
    assert type(docs[0].metadata["title"]) is str

from langchain.document_loaders.larksuite import LarkSuiteDocLoader

DOMAIN = ""
ACCESS_TOKEN = ""
DOCUMENT_ID = ""


def test_larksuite_doc_loader() -> None:
    """Test LarkSuite (FeiShu) document loader."""
    loader = LarkSuiteDocLoader(DOMAIN, ACCESS_TOKEN, DOCUMENT_ID)
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content is not None

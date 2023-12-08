from langchain.document_loaders.figma import FigmaFileLoader

ACCESS_TOKEN = ""
IDS = ""
KEY = ""


def test_figma_file_loader() -> None:
    """Test Figma file loader."""
    loader = FigmaFileLoader(ACCESS_TOKEN, IDS, KEY)
    docs = loader.load()

    assert len(docs) == 1

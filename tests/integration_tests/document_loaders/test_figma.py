from langchain.document_loaders.figma import FigmaFileLoader

ACCESS_TOKEN = "figd_Buj8TSTDYrvH1FwIzOqT6NUNEdZ5i1Cxo2J_tEPN"
IDS = "2:104"
KEY = "UXKwV2DyPaXCgBSufVIXZy"


def test_figma_file_loader() -> None:
    """Test Figma file loader."""
    loader = FigmaFileLoader(ACCESS_TOKEN, IDS, KEY)
    docs = loader.load()

    assert len(docs) == 1

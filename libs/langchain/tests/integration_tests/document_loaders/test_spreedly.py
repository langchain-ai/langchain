from langchain.document_loaders.spreedly import SpreedlyLoader


def test_spreedly_loader() -> None:
    """Test Spreedly Loader."""
    access_token = ""
    resource = "gateways_options"
    spreedly_loader = SpreedlyLoader(access_token, resource)
    documents = spreedly_loader.load()

    assert len(documents) == 1

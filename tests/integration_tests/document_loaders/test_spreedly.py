import StripeLoader

access_token = ""
resource = "gateways_options"

def test_spreedly_loader() -> None:
    """Test Spreedly oader."""
    spreedly_loader = SpreedlyLoader(access_token, resource)
    documents = spreedly_loader.load()

    assert len(documents) == 1

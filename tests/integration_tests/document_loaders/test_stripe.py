from langchain.document_loaders.stripe import StripeLoader

access_token = ""
resource = "charges"


def test_stripe_loader() -> None:
    """Test Figma file loader."""
    stripe_loader = StripeLoader(access_token, resource)
    documents = stripe_loader.load()

    assert len(documents) == 1

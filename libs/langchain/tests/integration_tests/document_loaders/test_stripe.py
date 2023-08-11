from langchain.document_loaders.stripe import StripeLoader


def test_stripe_loader() -> None:
    """Test Stripe file loader."""
    stripe_loader = StripeLoader("charges")
    documents = stripe_loader.load()

    assert len(documents) == 1

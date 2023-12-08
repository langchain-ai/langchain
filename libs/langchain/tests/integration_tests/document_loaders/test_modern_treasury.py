from langchain.document_loaders.modern_treasury import ModernTreasuryLoader


def test_modern_treasury_loader() -> None:
    """Test Modern Treasury file loader."""
    modern_treasury_loader = ModernTreasuryLoader("payment_orders")
    documents = modern_treasury_loader.load()

    assert len(documents) == 1

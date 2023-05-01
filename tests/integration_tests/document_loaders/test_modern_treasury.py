from langchain.document_loaders.modern_treasury import ModernTreasuryLoader

organization_id = ""
api_key = ""
resource = "payment_orders"


def test_modern_treasury_loader() -> None:
    """Test Modern Treasury file loader."""
    modern_treasury_loader = ModernTreasuryLoader(organization_id, api_key, resource)
    documents = modern_treasury_loader.load()

    assert len(documents) == 1

"""Integration test for the Steel Web Loader."""
import os
import pytest
from langchain_community.document_loaders import SteelWebLoader

requires_api_key = pytest.mark.skipif(
    not os.getenv("STEEL_API_KEY"),
    reason="Test requires STEEL_API_KEY environment variable"
)

@requires_api_key
def test_steel_loader_basic():
    """Test basic functionality of the Steel loader."""
    loader = SteelWebLoader("https://example.com")
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].page_content
    assert docs[0].metadata["source"] == "https://example.com"
    assert docs[0].metadata["steel_session_id"]
    assert docs[0].metadata["steel_session_viewer_url"]
    assert docs[0].metadata["extract_strategy"] == "text"

@requires_api_key
@pytest.mark.parametrize("strategy", ["text", "markdown", "html"])
def test_steel_loader_strategies(strategy):
    """Test different extraction strategies."""
    loader = SteelWebLoader(
        "https://example.com",
        extract_strategy=strategy
    )
    docs = loader.load()
    
    assert len(docs) == 1
    assert docs[0].metadata["extract_strategy"] == strategy

def test_steel_loader_invalid_strategy():
    """Test that invalid extraction strategy raises ValueError."""
    with pytest.raises(ValueError):
        SteelWebLoader("https://example.com", extract_strategy="invalid")

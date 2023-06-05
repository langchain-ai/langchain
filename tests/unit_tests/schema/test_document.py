"""Test document schema."""
from langchain.schema import Document


def test_document_hashes() -> None:
    """Test document hashing."""
    d = Document(page_content="hello")
    
    assert d.hash_ == "b1946ac92492d2347c6235b4d2611184"

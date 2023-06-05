"""Test document schema."""
from langchain.schema import Document


def test_document_hashes() -> None:
    """Test document hashing."""
    d1 = Document(page_content="hello")
    expected_hash = "0945717e-8d14-5f14-957f-0fb0ea1d56af"
    assert str(d1.hash_) == expected_hash

    d2 = Document(id="hello", page_content="hello")
    assert str(d2.hash_) == expected_hash

    d3 = Document(id="hello", page_content="hello2")
    assert str(d3.hash_) != expected_hash

    # Still fails. Need to update hash to hash metadata as well.
    d4 = Document(id="hello", page_content="hello", metadata={"source": "hello"})
    assert str(d4.hash_) != expected_hash

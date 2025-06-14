import pytest

from langchain_core.documents import Document
from langchain_core.indexing.api import _HashedDocument


def test_hashed_document_hashing() -> None:
    hashed_document = _HashedDocument(  # type: ignore[call-arg]
        uid="123", page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )
    assert isinstance(hashed_document.hash_, str)


def test_hashing_with_missing_content() -> None:
    """Check that ValueError is raised if page_content is missing."""
    with pytest.raises(TypeError):
        _HashedDocument(
            metadata={"key": "value"},
        )  # type: ignore[call-arg]


def test_uid_auto_assigned_to_hash() -> None:
    """Test uid is auto-assigned to the hashed_document hash."""
    hashed_document = _HashedDocument(  # type: ignore[call-arg]
        page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )
    assert hashed_document.uid == hashed_document.hash_


def test_to_document() -> None:
    """Test to_document method."""
    hashed_document = _HashedDocument(  # type: ignore[call-arg]
        page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )
    doc = hashed_document.to_document()
    assert isinstance(doc, Document)
    assert doc.page_content == "Lorem ipsum dolor sit amet"
    assert doc.metadata == {"key": "value"}


def test_from_document() -> None:
    """Test from document class method."""
    document = Document(
        page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )

    hashed_document = _HashedDocument.from_document(document)
    # hash should be deterministic
    assert hashed_document.hash_ == "fd1dc827-051b-537d-a1fe-1fa043e8b276"
    assert hashed_document.uid == hashed_document.hash_


def test_collection_hash_isolation():
    """
    Test that identical documents in different collections get different hashes.
    """
    # Same document content
    doc = Document(
        page_content="test content",
        metadata={"key": "value", "id": "123"},
    )

    # Create hashed documents for different collections
    hashed_doc_a = _HashedDocument.from_document(
        doc, collection_name="collection_a"
    )
    hashed_doc_b = _HashedDocument.from_document(
        doc, collection_name="collection_b"
    )

    # Assert they have different hashes and UIDs
    assert hashed_doc_a.hash_ != hashed_doc_b.hash_
    assert hashed_doc_a.uid != hashed_doc_b.uid

    # But same content and metadata
    assert hashed_doc_a.page_content == hashed_doc_b.page_content
    assert hashed_doc_a.metadata == hashed_doc_b.metadata


def test_collection_hash_same_collection():
    """Test that identical documents in the same collection get same hashes."""
    doc1 = Document(page_content="test", metadata={"id": "1"})
    doc2 = Document(page_content="test", metadata={"id": "1"})

    hashed_doc1 = _HashedDocument.from_document(doc1, collection_name="same_collection")
    hashed_doc2 = _HashedDocument.from_document(doc2, collection_name="same_collection")

    # Should have identical hashes (deduplication within same collection)
    assert hashed_doc1.hash_ == hashed_doc2.hash_
    assert hashed_doc1.uid == hashed_doc2.uid


def test_collection_hash_backward_compatibility():
    """Test that collection name defaults to empty string for backward compatibility."""
    doc = Document(page_content="test", metadata={"key": "value"})

    # Without collection name (old behavior)
    hashed_doc_old = _HashedDocument.from_document(doc)

    # With empty collection name (should be equivalent)
    hashed_doc_new = _HashedDocument.from_document(doc, collection_name="")

    assert hashed_doc_old.hash_ == hashed_doc_new.hash_
    assert hashed_doc_old.uid == hashed_doc_new.uid

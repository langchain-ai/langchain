import pytest

from langchain.indexes._api import _HashedDocument
from langchain.schema import Document


def test_hashed_document_hashing() -> None:
    hashed_document = _HashedDocument(
        uid="123", page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )
    assert isinstance(hashed_document.hash_, str)


def test_hashing_with_missing_content() -> None:
    """Check that ValueError is raised if page_content is missing."""
    with pytest.raises(ValueError):
        _HashedDocument(
            metadata={"key": "value"},
        )


def test_uid_auto_assigned_to_hash() -> None:
    """Test uid is auto-assigned to the hashed_document hash."""
    hashed_document = _HashedDocument(
        page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )
    assert hashed_document.uid == hashed_document.hash_


def test_to_document() -> None:
    """Test to_document method."""
    hashed_document = _HashedDocument(
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

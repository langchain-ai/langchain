import uuid
from typing import Literal

from langchain_core.documents import Document
from langchain_core.indexing.api import _get_document_with_hash


def test_hashed_document_hashing() -> None:
    document = Document(
        uid="123", page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )
    hashed_document = _get_document_with_hash(document, key_encoder="sha1")
    assert isinstance(hashed_document.id, str)


def test_to_document() -> None:
    """Test to_document method."""
    original_doc = Document(
        page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )
    hashed_doc = _get_document_with_hash(original_doc, key_encoder="sha1")
    assert isinstance(hashed_doc, Document)
    assert hashed_doc is not original_doc
    assert hashed_doc.page_content == "Lorem ipsum dolor sit amet"
    assert hashed_doc.metadata["key"] == "value"


def test_hashing() -> None:
    """Test from document class method."""
    document = Document(
        page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )
    hashed_document = _get_document_with_hash(document, key_encoder="sha1")
    # hash should be deterministic
    assert hashed_document.id == "fd1dc827-051b-537d-a1fe-1fa043e8b276"

    # Verify that hashing with sha1 is deterministic
    another_hashed_document = _get_document_with_hash(document, key_encoder="sha1")
    assert another_hashed_document.id == hashed_document.id

    # Verify that the result is different from SHA256, SHA512, blake2b
    values: list[Literal["sha256", "sha512", "blake2b"]] = [
        "sha256",
        "sha512",
        "blake2b",
    ]

    for key_encoder in values:
        different_hashed_document = _get_document_with_hash(
            document, key_encoder=key_encoder
        )
        assert different_hashed_document.id != hashed_document.id


def test_all_key_encoders_produce_uuids() -> None:
    """Every built-in key encoder must produce a valid UUID string.

    Vector stores like Qdrant validate that point IDs are UUIDs; raw hex
    digests from sha256/sha512/blake2b are not UUID-shaped and would be
    rejected at insert time.
    """
    document = Document(
        page_content="Lorem ipsum dolor sit amet", metadata={"key": "value"}
    )
    for key_encoder in ("sha1", "sha256", "sha512", "blake2b"):
        hashed = _get_document_with_hash(document, key_encoder=key_encoder)
        # Must parse as a UUID (raises ValueError otherwise).
        parsed = uuid.UUID(hashed.id)
        assert str(parsed) == hashed.id


def test_hashing_custom_key_encoder() -> None:
    """Test hashing with a custom key encoder."""

    def custom_key_encoder(doc: Document) -> str:
        return f"quack-{doc.metadata['key']}"

    document = Document(
        page_content="Lorem ipsum dolor sit amet", metadata={"key": "like a duck"}
    )
    hashed_document = _get_document_with_hash(document, key_encoder=custom_key_encoder)
    assert hashed_document.id == "quack-like a duck"
    assert isinstance(hashed_document.id, str)

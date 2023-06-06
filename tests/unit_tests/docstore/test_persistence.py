from langchain.docstore.artifacts import serialize_document, deserialize_document
from langchain.schema import Document


def test_serialization() -> None:
    """Test serialization."""
    initial_doc = Document(page_content="hello")
    serialized_doc = serialize_document(initial_doc)
    assert isinstance(serialized_doc, str)
    deserialized_doc = deserialize_document(serialized_doc)
    assert isinstance(deserialized_doc, Document)
    assert deserialized_doc == initial_doc


def test_serialization_with_metadata() -> None:
    """Test serialization with metadata."""
    initial_doc = Document(page_content="hello", metadata={"source": "hello"})
    serialized_doc = serialize_document(initial_doc)
    assert isinstance(serialized_doc, str)
    deserialized_doc = deserialize_document(serialized_doc)
    assert isinstance(deserialized_doc, Document)
    assert deserialized_doc == initial_doc

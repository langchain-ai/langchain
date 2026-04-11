from langchain_core.documents import Document


def test_init() -> None:
    for doc in [
        Document(page_content="foo"),
        Document(page_content="foo", metadata={"a": 1}),
        Document(page_content="foo", id=None),
        Document(page_content="foo", id="1"),
        Document(page_content="foo", id=1),
    ]:
        assert isinstance(doc, Document)

def test_blob_as_bytes_io_with_string_data() -> None:
    """Test that as_bytes_io() works correctly with string data."""
    from langchain_core.documents.base import Blob

    blob = Blob.from_data("hello")
    with blob.as_bytes_io() as f:
        assert f.read() == b"hello"

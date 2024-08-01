from langchain_core.documents import Document


def test_str() -> None:
    assert str(Document(page_content="Hello, World!")) == "page_content='Hello, World!'"
    assert (
        str(Document(page_content="Hello, World!", metadata={"a": 3}))
        == "page_content='Hello, World!' metadata={'a': 3}"
    )


def test_repr() -> None:
    assert (
        repr(Document(page_content="Hello, World!"))
        == "Document(page_content='Hello, World!')"
    )
    assert (
        repr(Document(page_content="Hello, World!", metadata={"a": 3}))
        == "Document(metadata={'a': 3}, page_content='Hello, World!')"
    )

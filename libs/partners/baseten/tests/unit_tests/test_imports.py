"""Test imports."""


def test_import_chat_model() -> None:
    """Test that we can import the chat model."""
    from langchain_baseten import ChatBaseten  # noqa: F401


def test_import_embeddings() -> None:
    """Test that we can import the embeddings model."""
    from langchain_baseten import BasetenEmbeddings  # noqa: F401


def test_import_all() -> None:
    """Test that we can import all from the package."""
    from langchain_baseten import __all__

    assert "ChatBaseten" in __all__
    assert "BasetenEmbeddings" in __all__

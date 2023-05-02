from langchain.document_loaders.parsers import __all__


def test_public_api() -> None:
    """Simple test to verify that the public API wasn't broken."""
    assert __all__ == ["BaseBlobParser"]

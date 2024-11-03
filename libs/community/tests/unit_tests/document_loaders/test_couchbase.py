"""Test importing the Couchbase document loader."""


def test_couchbase_import() -> None:
    """Test that the Couchbase document loader can be imported."""
    from langchain_community.document_loaders import CouchbaseLoader  # noqa: F401

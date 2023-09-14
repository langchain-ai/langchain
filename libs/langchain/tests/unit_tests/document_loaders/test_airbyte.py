"""Test the airbyte document loader.

Light test to ensure that the airbyte document loader can be imported.
"""


def test_airbyte_import() -> None:
    """Test that the airbyte document loader can be imported."""
    from langchain.document_loaders import airbyte  # noqa

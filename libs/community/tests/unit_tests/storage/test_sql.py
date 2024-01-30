"""Light weight unit test that attempts to import SQLDocStore/SQLStrStore.
"""


def test_import_storage() -> None:
    """Attempt to import storage modules."""
    from langchain_community.storage.sql import SQLDocStore, SQLStrStore  # noqa

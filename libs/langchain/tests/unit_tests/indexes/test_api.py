from langchain.indexes import __all__


def test_all() -> None:
    """Use to catch obvious breaking changes."""
    expected = [
        "aindex",
        "GraphIndexCreator",
        "index",
        "IndexingResult",
        "SQLRecordManager",
        "VectorstoreIndexCreator",
    ]
    assert __all__ == sorted(expected, key=lambda x: x.lower())

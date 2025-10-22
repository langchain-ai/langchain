from langchain_classic.indexes import __all__


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
    assert sorted(__all__) == sorted(expected)

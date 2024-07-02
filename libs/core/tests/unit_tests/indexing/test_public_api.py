from langchain_core.indexing import __all__


def test_all() -> None:
    """Use to catch obvious breaking changes."""
    assert __all__ == sorted(__all__, key=str.lower)
    assert __all__ == [
        "aindex",
        "index",
        "IndexingResult",
        "InMemoryRecordManager",
        "RecordManager",
        "UpsertResponse",
    ]

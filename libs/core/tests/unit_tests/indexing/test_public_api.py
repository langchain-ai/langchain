from langchain_core.indexing import __all__


def test_all() -> None:
    """Use to catch obvious breaking changes."""
    assert list(__all__) == sorted(__all__, key=str.lower)
    assert set(__all__) == {
        "aindex",
        "DeleteResponse",
        "DocumentIndex",
        "index",
        "IndexingResult",
        "InMemoryRecordManager",
        "RecordManager",
        "UpsertResponse",
    }

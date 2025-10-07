from langchain_classic.indexes import __all__

EXPECTED_ALL = [
    # Keep sorted
    "aindex",
    "GraphIndexCreator",
    "index",
    "IndexingResult",
    "SQLRecordManager",
    "VectorstoreIndexCreator",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

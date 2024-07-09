from langchain_couchbase import __all__

EXPECTED_ALL = [
    "CouchbaseVectorStore",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

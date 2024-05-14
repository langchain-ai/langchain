from langchain import docstore

EXPECTED_ALL = ["DocstoreFn", "InMemoryDocstore", "Wikipedia"]


def test_all_imports() -> None:
    assert set(docstore.__all__) == set(EXPECTED_ALL)

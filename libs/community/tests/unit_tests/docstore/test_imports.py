from langchain_community.docstore import __all__

EXPECTED_ALL = ["DocstoreFn", "InMemoryDocstore", "Wikipedia"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

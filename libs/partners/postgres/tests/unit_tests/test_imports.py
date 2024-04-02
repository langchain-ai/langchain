from langchain_postgres import __all__

EXPECTED_ALL = ["__version__", "PostgresChatMessageHistory"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

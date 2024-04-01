from langchain_postgres import __all__

EXPECTED_ALL = [
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

from langchain_astraflow import __all__

EXPECTED_ALL = ["ChatAstraflow"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

from langchain_airbyte import __all__

EXPECTED_ALL = [
    "AirbyteLoader",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

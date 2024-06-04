from langchain_nomic import __all__

EXPECTED_ALL = [
    "NomicEmbeddings",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

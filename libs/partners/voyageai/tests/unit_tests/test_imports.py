from langchain_voyageai import __all__

EXPECTED_ALL = [
    "VoyageAIEmbeddings",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

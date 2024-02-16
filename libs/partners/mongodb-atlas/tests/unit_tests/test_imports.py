from langchain_mongodb_atlas import __all__

EXPECTED_ALL = [
    "MongoDBAtlasVectorSearch",
    "MongoDBAtlasVectorSearch",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

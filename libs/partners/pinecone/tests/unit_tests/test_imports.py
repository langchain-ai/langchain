from langchain_pinecone import __all__

EXPECTED_ALL = [
    "PineconeVectorStore",
    "Pinecone",
    "PineconeEmbeddings",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

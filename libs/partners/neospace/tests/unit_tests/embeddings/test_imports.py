from langchain_neospace.embeddings import __all__

EXPECTED_ALL = ["OpenAIEmbeddings", "AzureOpenAIEmbeddings"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

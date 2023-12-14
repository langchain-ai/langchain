from langchain_nvidia_aiplay import __all__

EXPECTED_ALL = ["ChatNVAIPlay", "NVAIPlayEmbeddings"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

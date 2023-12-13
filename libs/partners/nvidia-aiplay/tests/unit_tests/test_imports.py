from nvidia_aiplay import __all__

EXPECTED_ALL = ["NVAIPlayLLM", "ChatNVAIPlay", "NVAIPlayVectorStore"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

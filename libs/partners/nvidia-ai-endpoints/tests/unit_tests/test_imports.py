from langchain_nvidia_ai_endpoints import __all__

EXPECTED_ALL = ["ChatNVIDIA", "NVIDIAEmbeddings"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

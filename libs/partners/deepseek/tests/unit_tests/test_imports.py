from langchain_deepseek import __all__

EXPECTED_ALL = ["__version__", "ChatDeepSeek"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)

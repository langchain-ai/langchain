from langchain_classic.prompts.prompt import __all__

EXPECTED_ALL = ["Prompt", "PromptTemplate"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

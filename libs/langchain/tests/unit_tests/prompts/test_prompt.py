from langchain.prompts.prompt import __all__

EXPECTED_ALL = ["Prompt", "PromptTemplate"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

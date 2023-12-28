from langchain.schema.prompt import __all__

EXPECTED_ALL = ["PromptValue"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

from langchain.schema.prompt_template import __all__

EXPECTED_ALL = ["BasePromptTemplate", "format_document"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

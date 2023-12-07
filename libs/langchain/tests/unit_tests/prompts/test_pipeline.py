from langchain.prompts.pipeline import __all__

EXPECTED_ALL = ["PipelinePromptTemplate", "_get_inputs"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

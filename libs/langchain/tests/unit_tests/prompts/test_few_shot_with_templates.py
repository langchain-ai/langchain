from langchain_classic.prompts.few_shot_with_templates import __all__

EXPECTED_ALL = ["FewShotPromptWithTemplates"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

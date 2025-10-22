from langchain_classic.prompts.few_shot import __all__

EXPECTED_ALL = [
    "FewShotChatMessagePromptTemplate",
    "FewShotPromptTemplate",
    "_FewShotPromptTemplateMixin",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

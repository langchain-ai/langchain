from langchain import prompts

EXPECTED_ALL = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "HumanMessagePromptTemplate",
    "LengthBasedExampleSelector",
    "MaxMarginalRelevanceExampleSelector",
    "MessagesPlaceholder",
    "PromptTemplate",
    "SemanticSimilarityExampleSelector",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "FewShotChatMessagePromptTemplate",
]


def test_all_imports() -> None:
    assert set(prompts.__all__) == set(EXPECTED_ALL)

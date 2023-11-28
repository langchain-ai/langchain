from langchain_core.prompts import __all__

EXPECTED_ALL = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BaseMessagePromptTemplate",
    "BasePromptTemplate",
    "BaseStringMessagePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "FewShotChatMessagePromptTemplate",
    "HumanMessagePromptTemplate",
    "MessageLike",
    "MessageLikeRepresentation",
    "MessagesPlaceholder",
    "MessagePromptTemplateT",
    "PipelinePromptTemplate",
    "PromptTemplate",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "load_prompt",
    "format_document",
    "check_valid_template",
    "get_template_variables",
    "jinja2_formatter",
    "validate_jinja2",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

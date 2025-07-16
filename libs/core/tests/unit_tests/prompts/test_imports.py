from langchain_core.prompts import __all__

EXPECTED_ALL = [
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "DictPromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    "FewShotChatMessagePromptTemplate",
    "format_document",
    "aformat_document",
    "HumanMessagePromptTemplate",
    "MessagesPlaceholder",
    "PipelinePromptTemplate",
    "PromptTemplate",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    "load_prompt",
    "check_valid_template",
    "get_template_variables",
    "jinja2_formatter",
    "validate_jinja2",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

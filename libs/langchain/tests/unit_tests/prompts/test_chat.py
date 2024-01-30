from langchain.prompts.chat import __all__

EXPECTED_ALL = [
    "MessageLike",
    "MessageLikeRepresentation",
    "MessagePromptTemplateT",
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BaseMessagePromptTemplate",
    "BaseStringMessagePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    "ChatPromptValue",
    "ChatPromptValueConcrete",
    "HumanMessagePromptTemplate",
    "MessagesPlaceholder",
    "SystemMessagePromptTemplate",
    "_convert_to_message",
    "_create_template_from_message_type",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)

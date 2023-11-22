from langchain_core.prompt_values import (
    ChatPromptValue,
    ChatPromptValueConcrete,
    StringPromptValue,
)


def test_lc_namespace() -> None:
    assert StringPromptValue.get_lc_namespace() == ["langchain", "prompts", "base"]
    assert ChatPromptValue.get_lc_namespace() == ["langchain", "prompts", "chat"]
    assert ChatPromptValueConcrete.get_lc_namespace() == [
        "langchain",
        "prompts",
        "chat",
    ]

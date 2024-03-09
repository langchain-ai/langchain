from langchain_core.prompts import ChatPromptTemplate

from langchain import hub


def test_hub_pull_public_prompt() -> None:
    prompt = hub.pull("efriis/my-first-prompt")
    assert isinstance(prompt, ChatPromptTemplate)

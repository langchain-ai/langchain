from langchain_core.prompts import ChatPromptTemplate

from langchain import hub


def test_hub_pull_public_prompt() -> None:
    prompt = hub.pull("efriis/my-first-prompt")
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.metadata is not None
    assert prompt.metadata["lc_hub_owner"] == "efriis"
    assert prompt.metadata["lc_hub_repo"] == "my-first-prompt"
    assert (
        prompt.metadata["lc_hub_commit_hash"]
        == "52668c2f392f8f52d2fc0d6b60cb964e3961934fdbd5dbe72b62926be6b51742"
    )

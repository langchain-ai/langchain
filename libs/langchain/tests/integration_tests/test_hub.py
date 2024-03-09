from langchain_core.prompts import ChatPromptTemplate

from langchain import hub


def test_hub_pull_public_prompt() -> None:
    prompt = hub.pull("efriis/my-first-prompt")
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.metadata is not None
    assert "lc_hub" in prompt.metadata
    hub_md = prompt.metadata["lc_hub"]
    assert hub_md
    assert hub_md["owner"] == "efriis"
    assert hub_md["repo"] == "my-first-prompt"
    assert (
        hub_md["commit_hash"]
        == "52668c2f392f8f52d2fc0d6b60cb964e3961934fdbd5dbe72b62926be6b51742"
    )

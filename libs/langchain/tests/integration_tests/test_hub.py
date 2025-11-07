import os

from langchain_core.prompts import ChatPromptTemplate

from langchain_classic import hub


def test_hub_pull_public_prompt() -> None:
    prompt = hub.pull("efriis/my-first-prompt")
    assert isinstance(prompt, ChatPromptTemplate)
    assert prompt.metadata is not None
    assert prompt.metadata["lc_hub_owner"] == "efriis"
    assert prompt.metadata["lc_hub_repo"] == "my-first-prompt"
    assert (
        prompt.metadata["lc_hub_commit_hash"]
        == "56489e79537fc477d8368e6c9902df15b5e9fe8bc0e4f38dc4b15b65e550077c"
    )


def test_hub_pull_private_prompt() -> None:
    private_prompt = hub.pull("integration-test", api_key=os.environ["HUB_API_KEY"])
    assert isinstance(private_prompt, ChatPromptTemplate)
    assert private_prompt.metadata is not None
    assert private_prompt.metadata["lc_hub_owner"] == "-"
    assert private_prompt.metadata["lc_hub_repo"] == "integration-test"

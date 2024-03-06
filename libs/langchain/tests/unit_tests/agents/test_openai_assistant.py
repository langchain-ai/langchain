import pytest

from langchain.agents.openai_assistant import OpenAIAssistantRunnable


@pytest.mark.requires("openai")
def test_user_supplied_client() -> None:
    import openai

    client = openai.AzureOpenAI(
        azure_endpoint="azure_endpoint",
        api_key="api_key",
        api_version="api_version",
    )

    assistant = OpenAIAssistantRunnable(
        assistant_id="assistant_id",
        client=client,
    )

    assert assistant.client == client

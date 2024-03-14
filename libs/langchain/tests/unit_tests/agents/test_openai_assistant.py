from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from langchain.agents.openai_assistant import OpenAIAssistantRunnable


def _create_mock_client(*args, **kwargs) -> Any:
    client = MagicMock()
    client.beta.assistants.create().id = "abc123"
    return client


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


@patch(
    "langchain.agents.openai_assistant.base._get_openai_client",
    new=_create_mock_client,
)
def test_create_assistant() -> None:
    assistant = OpenAIAssistantRunnable.create_assistant(
        name="name",
        instructions="instructions",
        tools=[{"type": "code_interpreter"}],
        model="",
    )
    assert isinstance(assistant, OpenAIAssistantRunnable)

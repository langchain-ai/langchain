from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain.agents.openai_assistant import OpenAIAssistantRunnable


def _create_mock_client(*args: Any, use_async: bool = False, **kwargs: Any) -> Any:
    client = AsyncMock() if use_async else MagicMock()
    mock_assistant = MagicMock()
    mock_assistant.id = "abc123"
    client.beta.assistants.create.return_value = mock_assistant
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


@pytest.mark.requires("openai")
@patch(
    "langchain.agents.openai_assistant.base._get_openai_client",
    new=partial(_create_mock_client, use_async=False),
)
def test_create_assistant() -> None:
    assistant = OpenAIAssistantRunnable.create_assistant(
        name="name",
        instructions="instructions",
        tools=[{"type": "code_interpreter"}],
        model="",
    )
    assert isinstance(assistant, OpenAIAssistantRunnable)


@pytest.mark.requires("openai")
@patch(
    "langchain.agents.openai_assistant.base._get_openai_async_client",
    new=partial(_create_mock_client, use_async=True),
)
async def test_acreate_assistant() -> None:
    assistant = await OpenAIAssistantRunnable.acreate_assistant(
        name="name",
        instructions="instructions",
        tools=[{"type": "code_interpreter"}],
        model="",
        client=_create_mock_client(),
    )
    assert isinstance(assistant, OpenAIAssistantRunnable)

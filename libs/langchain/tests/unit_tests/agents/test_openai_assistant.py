from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_classic.agents.openai_assistant import OpenAIAssistantRunnable


def _create_mock_client(*_: Any, use_async: bool = False, **__: Any) -> Any:
    client = AsyncMock() if use_async else MagicMock()
    mock_assistant = MagicMock()
    mock_assistant.id = "abc123"
    client.beta.assistants.create.return_value = mock_assistant
    return client


@pytest.mark.requires("openai")
def test_user_supplied_client() -> None:
    openai = pytest.importorskip("openai")

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
async def test_ainvoke_uses_async_response_completed() -> None:
    # Arrange a runner with mocked async client and a completed run
    assistant = OpenAIAssistantRunnable(
        assistant_id="assistant_id",
        client=_create_mock_client(),
        async_client=_create_mock_client(use_async=True),
        as_agent=False,
    )
    mock_run = MagicMock()
    mock_run.id = "run-id"
    mock_run.thread_id = "thread-id"
    mock_run.status = "completed"

    # await_for_run returns a completed run
    await_for_run_mock = AsyncMock(return_value=mock_run)
    # async messages list returns messages belonging to run
    msg = MagicMock()
    msg.run_id = "run-id"
    msg.content = []
    list_mock = AsyncMock(return_value=[msg])

    with (
        patch.object(assistant, "_await_for_run", await_for_run_mock),
        patch.object(
            assistant.async_client.beta.threads.messages,
            "list",
            list_mock,
        ),
    ):
        # Act
        result = await assistant.ainvoke({"content": "hi"})

    # Assert: returns messages list (non-agent path) and did not block
    assert isinstance(result, list)
    list_mock.assert_awaited()


@pytest.mark.requires("openai")
@patch(
    "langchain.agents.openai_assistant.base._get_openai_async_client",
    new=partial(_create_mock_client, use_async=True),
)
async def test_ainvoke_uses_async_response_requires_action_agent() -> None:
    # Arrange a runner with mocked async client and requires_action run
    assistant = OpenAIAssistantRunnable(
        assistant_id="assistant_id",
        client=_create_mock_client(),
        async_client=_create_mock_client(use_async=True),
        as_agent=True,
    )
    mock_run = MagicMock()
    mock_run.id = "run-id"
    mock_run.thread_id = "thread-id"
    mock_run.status = "requires_action"

    # Fake tool call structure
    tool_call = MagicMock()
    tool_call.id = "tool-id"
    tool_call.function.name = "foo"
    tool_call.function.arguments = '{\n  "x": 1\n}'
    mock_run.required_action.submit_tool_outputs.tool_calls = [tool_call]

    await_for_run_mock = AsyncMock(return_value=mock_run)

    # Act
    with patch.object(assistant, "_await_for_run", await_for_run_mock):
        result = await assistant.ainvoke({"content": "hi"})

    # Assert: returns list of OpenAIAssistantAction
    assert isinstance(result, list)
    assert result
    assert getattr(result[0], "tool", None) == "foo"


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

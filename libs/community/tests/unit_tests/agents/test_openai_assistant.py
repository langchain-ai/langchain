from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_community.agents.openai_assistant import OpenAIAssistantV2Runnable


def _create_mock_client(*args: Any, use_async: bool = False, **kwargs: Any) -> Any:
    client = AsyncMock() if use_async else MagicMock()
    client.beta.threads.runs.create = MagicMock(return_value=None)
    return client


@pytest.mark.requires("openai")
def test_set_run_truncation_params() -> None:
    client = _create_mock_client()

    assistant = OpenAIAssistantV2Runnable(assistant_id="assistant_xyz", client=client)
    input = {
        "content": "AI question",
        "thread_id": "thread_xyz",
        "instructions": "You're a helpful assistant; answer questions as best you can.",
        "model": "gpt-4o",
        "max_prompt_tokens": 2000,
        "truncation_strategy": {"type": "last_messages", "last_messages": 10},
    }
    expected_response = {
        "assistant_id": "assistant_xyz",
        "instructions": "You're a helpful assistant; answer questions as best you can.",
        "model": "gpt-4o",
        "max_prompt_tokens": 2000,
        "truncation_strategy": {"type": "last_messages", "last_messages": 10},
    }

    assistant._create_run(input=input)
    _, kwargs = client.beta.threads.runs.create.call_args

    assert kwargs == expected_response

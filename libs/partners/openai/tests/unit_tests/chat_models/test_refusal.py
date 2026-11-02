"""Replay standardized refusals through OpenAI APIs."""

import pytest
from langchain_core.messages import AIMessage

from langchain_openai.chat_models._compat import _convert_from_v1_to_chat_completions
from langchain_openai.chat_models.base import _construct_responses_api_input


@pytest.mark.parametrize("native", [False, True])
def test_refusal_responses_replay(*, native: bool) -> None:
    message = AIMessage(
        content=[
            {
                "type": "refusal",
                "refusal": "blocked",
                "id": "msg_1",
                **(
                    {"phase": "final_answer"}
                    if native
                    else {"extras": {"phase": "final_answer"}}
                ),
            }
        ]
    )
    result = _construct_responses_api_input([message])
    assert result == [
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "phase": "final_answer",
            "content": [{"type": "refusal", "refusal": "blocked"}],
        }
    ]


def test_refusal_chat_completions_replay() -> None:
    message = AIMessage(
        content_blocks=[
            {
                "type": "refusal",
                "refusal": "blocked",
                "extras": {"phase": "final_answer"},
            }
        ]
    )
    assert _convert_from_v1_to_chat_completions(message).content == [
        {"type": "refusal", "refusal": "blocked"}
    ]

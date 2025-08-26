from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.responses import ToolOutput

from tests.utils import BaseSchema, load_spec

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    skip_openai_integration_tests = True
else:
    skip_openai_integration_tests = False

AGENT_PROMPT = """
You are a strict polling bot.

- Only use the "poll_job" tool until it returns { status: "succeeded" }.
- If status is "pending", call the tool again. Do not produce a final answer.
- When it is "succeeded", return exactly: "Attempts: <number>" with no extra text.
"""


class TestCase(BaseSchema):
    name: str
    return_direct: bool
    response_format: Optional[Dict[str, Any]]
    expected_tool_calls: int
    expected_last_message: str
    expected_structured_response: Optional[Dict[str, Any]]


TEST_CASES = load_spec("return_direct", as_model=TestCase)


def _make_tool(return_direct: bool):
    attempts = 0

    def _side_effect():
        nonlocal attempts
        attempts += 1
        return {
            "status": "succeeded" if attempts >= 10 else "pending",
            "attempts": attempts,
        }

    mock = MagicMock(side_effect=_side_effect)

    @tool(
        "pollJob",
        description=(
            "Check the status of a long-running job. "
            "Returns { status: 'pending' | 'succeeded', attempts: number }."
        ),
        return_direct=return_direct,
    )
    def _wrapped():
        return mock()

    return {"tool": _wrapped, "mock": mock}


@pytest.mark.skipif(
    skip_openai_integration_tests, reason="OpenAI integration tests are disabled."
)
@pytest.mark.parametrize("case", TEST_CASES, ids=[c.name for c in TEST_CASES])
def test_return_direct_integration_matrix(case: TestCase) -> None:
    poll_tool = _make_tool(case.return_direct)

    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    if case.response_format:
        agent = create_react_agent(
            model,
            tools=[poll_tool["tool"]],
            prompt=AGENT_PROMPT,
            response_format=ToolOutput(case.response_format),
        )
    else:
        agent = create_react_agent(
            model,
            tools=[poll_tool["tool"]],
            prompt=AGENT_PROMPT,
        )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "Poll the job until it's done and tell me how many attempts it took."
                )
            ]
        }
    )

    # Count tool calls
    assert poll_tool["mock"].call_count == case.expected_tool_calls

    # Check last message content
    last_message = result["messages"][-1]
    assert last_message.content == case.expected_last_message

    # Check structured response
    if case.expected_structured_response is not None:
        structured_response_json = result["structured_response"]
        assert structured_response_json == case.expected_structured_response
    else:
        assert "structured_response" not in result

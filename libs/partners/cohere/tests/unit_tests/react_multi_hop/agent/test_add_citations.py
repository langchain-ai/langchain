from typing import Any, Dict
from unittest import mock

import pytest
from langchain_core.agents import AgentAction, AgentFinish

from langchain_cohere import CohereCitation
from langchain_cohere.react_multi_hop.agent import _AddCitations

CITATIONS = [CohereCitation(start=1, end=2, text="foo", documents=[{"bar": "baz"}])]
GENERATION = "mocked generation"


@pytest.mark.parametrize(
    "invoke_with,expected",
    [
        pytest.param({}, [], id="no agent_steps or chain_input"),
        pytest.param(
            {
                "chain_input": {"intermediate_steps": []},
                "agent_steps": [
                    AgentAction(
                        tool="tool_name", tool_input="tool_input", log="tool_log"
                    )
                ],
            },
            [AgentAction(tool="tool_name", tool_input="tool_input", log="tool_log")],
            id="not an AgentFinish",
        ),
        pytest.param(
            {
                "chain_input": {
                    "intermediate_steps": [
                        (
                            AgentAction(
                                tool="tool_name",
                                tool_input="tool_input",
                                log="tool_log",
                            ),
                            {"tool_output": "output"},
                        )
                    ]
                },
                "agent_steps": AgentFinish(
                    return_values={"output": "output1", "grounded_answer": GENERATION},
                    log="",
                ),
            },
            AgentFinish(
                return_values={"output": GENERATION, "citations": CITATIONS}, log=""
            ),
            id="AgentFinish",
        ),
    ],
)
@mock.patch(
    "langchain_cohere.react_multi_hop.agent.parse_citations",
    autospec=True,
    return_value=(GENERATION, CITATIONS),
)
def test_add_citations(
    parse_citations_mock: Any, invoke_with: Dict[str, Any], expected: Any
) -> None:
    chain = _AddCitations()
    actual = chain.invoke(invoke_with)

    assert expected == actual

    if isinstance(expected, AgentFinish):
        parse_citations_mock.assert_called_once_with(
            grounded_answer=GENERATION, documents=[{"tool_output": "output"}]
        )

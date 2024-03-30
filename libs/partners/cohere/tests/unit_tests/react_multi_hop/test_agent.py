from typing import List, Tuple

import pytest
from langchain_core.agents import AgentAction

from langchain_cohere.react_multi_hop.agent import render_intermediate_steps


@pytest.mark.parametrize(
    "intermediate_steps,expected",
    [
        pytest.param([], "", id="no steps"),
        pytest.param(
            [
                (
                    AgentAction(
                        tool="tool_1", tool_input="tool_1_input", log="tool_1_log"
                    ),
                    "observation_1",
                )
            ],
            "",
            id="single step",
        ),
    ],
)
def test_render_intermediate_steps(
    intermediate_steps: List[Tuple[AgentAction, str]], expected: str
) -> None:
    actual = render_intermediate_steps(intermediate_steps)

    assert expected == actual

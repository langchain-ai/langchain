from typing import Any, Dict, List
from unittest import mock

import pytest
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.messages import AIMessage

from langchain_cohere.react_multi_hop.parsing import CohereToolsReactAgentOutputParser
from tests.unit_tests.react_multi_hop import ExpectationType, read_expectation_from_file


@pytest.mark.parametrize(
    "scenario_name,expected",
    [
        pytest.param(
            "answer_sound_of_music",
            AgentFinish(
                return_values={
                    "output": "Best Buy, originally called Sound of Music, was added to Standard & Poor's S&P 500 in 1999.",  # noqa: E501
                    "grounded_answer": "<co: 0,2>Best Buy</co: 0,2>, originally called Sound of Music, was added to <co: 2>Standard & Poor's S&P 500</co: 2> in <co: 2>1999</co: 2>.",  # noqa: E501
                },
                log="Relevant Documents: 0,2,3\nCited Documents: 0,2\nAnswer: Best Buy, originally called Sound of Music, was added to Standard & Poor's S&P 500 in 1999.\nGrounded answer: <co: 0,2>Best Buy</co: 0,2>, originally called Sound of Music, was added to <co: 2>Standard & Poor's S&P 500</co: 2> in <co: 2>1999</co: 2>.",  # noqa: E501
            ),
            id="best buy example",
        )
    ],
)
def test_it_parses_answer(scenario_name: str, expected: AgentFinish) -> None:
    text = read_expectation_from_file(ExpectationType.completions, scenario_name)
    actual = CohereToolsReactAgentOutputParser().parse(text)

    assert expected == actual


@mock.patch("langchain_cohere.react_multi_hop.parsing.parse_actions", autospec=True)
def test_it_returns_parses_action(parse_actions_mock: mock.Mock) -> None:
    # The actual parsing is mocked and tested elsewhere
    text = "Reflection: mocked"
    generation = "mocked generation"
    plan = "mocked plan"
    parser = CohereToolsReactAgentOutputParser()
    parsed_actions: List[Dict[str, Any]] = [
        {"tool_name": "tool1", "parameters": {"param1": "value1"}},
        {"tool_name": "tool2", "parameters": {"param2": "value2"}},
    ]
    parse_actions_mock.return_value = (generation, plan, parsed_actions)
    expected = [
        AgentActionMessageLog(
            tool=parsed_actions[0]["tool_name"],
            tool_input=parsed_actions[0]["parameters"],
            log=f"\n{plan}\n{str(parsed_actions[0])}\n",
            message_log=[AIMessage(content=generation)],
        ),
        AgentActionMessageLog(
            tool=parsed_actions[1]["tool_name"],
            tool_input=parsed_actions[1]["parameters"],
            log=f"\n{str(parsed_actions[1])}\n",
            message_log=[AIMessage(content=generation)],
        ),
    ]

    actual = parser.parse(text)

    parse_actions_mock.assert_called_once_with(text)
    assert expected == actual

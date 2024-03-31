from typing import Any

import pytest
from langchain_core.agents import AgentFinish
from langchain_core.messages import SystemMessage

from langchain_cohere.react_multi_hop.agent import (
    CohereToolsReactAgentOutputParser,
    render_observations,
)

COMPLETIONS = [
    """Relevant Documents: 0,2,3
Cited Documents: 0,2
Answer: Best Buy, originally called Sound of Music, was added to Standard & Poor's S&P 500 in 1999.
Grounded answer: <co: 0,2>Best Buy</co: 0,2>, originally called Sound of Music, was added to <co: 2>Standard & Poor's S&P 500</co: 2> in <co: 2>1999</co: 2>."""  # noqa: E501
]


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param(
            COMPLETIONS[0],
            AgentFinish(
                return_values={
                    "output": "Best Buy, originally called Sound of Music, was added to Standard & Poor's S&P 500 in 1999."  # noqa: E501
                },
                log="Relevant Documents: 0,2,3\nCited Documents: 0,2\nAnswer: Best Buy, originally called Sound of Music, was added to Standard & Poor's S&P 500 in 1999.\nGrounded answer: <co: 0,2>Best Buy</co: 0,2>, originally called Sound of Music, was added to <co: 2>Standard & Poor's S&P 500</co: 2> in <co: 2>1999</co: 2>.",  # noqa: E501
            ),
            id="best buy example",
        )
    ],
)
def test_parse_agent_finish(text: str, expected: AgentFinish) -> None:
    actual = CohereToolsReactAgentOutputParser().parse(text)

    assert expected == actual


document_template = """Document: {index}
{fields}"""


@pytest.mark.parametrize(
    "observation,expected_content",
    [
        pytest.param(
            "foo", document_template.format(index=0, fields="Output: foo"), id="string"
        ),
        pytest.param(
            {"foo": "bar"},
            document_template.format(index=0, fields="Foo: bar"),
            id="dictionary",
        ),
        pytest.param(
            {"url": "foo"},
            document_template.format(index=0, fields="URL: foo"),
            id="dictionary with url",
        ),
        pytest.param(
            {"foo": "bar", "baz": "foobar"},
            document_template.format(index=0, fields="Foo: bar\nBaz: foobar"),
            id="dictionary with multiple keys",
        ),
        pytest.param(
            ["foo", "bar"],
            "\n\n".join(
                [
                    document_template.format(index=0, fields="Output: foo"),
                    document_template.format(index=1, fields="Output: bar"),
                ]
            ),
            id="list of strings",
        ),
        pytest.param(
            [{"foo": "bar"}, {"baz": "foobar"}],
            "\n\n".join(
                [
                    document_template.format(index=0, fields="Foo: bar"),
                    document_template.format(index=1, fields="Baz: foobar"),
                ]
            ),
            id="list of dictionaries",
        ),
    ],
)
def test_render_observation_has_correct_content(
    observation: Any, expected_content: str
) -> None:
    actual, _ = render_observations(observations=observation, index=0)
    expected_content = f"<results>\n{expected_content}\n</results>"

    assert SystemMessage(content=expected_content) == actual


def test_render_observation_has_correct_indexes() -> None:
    index = 13
    observations = ["foo", "bar"]
    expected_index = 15

    _, actual = render_observations(observations=observations, index=index)

    assert expected_index == actual

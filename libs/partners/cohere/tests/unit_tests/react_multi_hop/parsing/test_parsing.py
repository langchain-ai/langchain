import pytest
from langchain_core.agents import AgentFinish

from langchain_cohere.react_multi_hop.agent import CohereToolsReactAgentOutputParser

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

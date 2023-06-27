"""Test agent trajectory evaluation chain."""

from typing import List, Tuple

import pytest

from langchain.evaluation.agents.trajectory_eval_chain import TrajectoryEvalChain
from langchain.schema import AgentAction
from langchain.tools.base import tool
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.fixture
def intermediate_steps() -> List[Tuple[AgentAction, str]]:
    return [
        (
            AgentAction(
                tool="Foo",
                tool_input="Bar",
                log="Star date 2021-06-13: Foo received input: Bar",
            ),
            "Baz",
        ),
    ]


@tool
def foo(bar: str) -> str:
    """Foo."""
    return bar


def test_trajectory_eval_chain(
    intermediate_steps: List[Tuple[AgentAction, str]]
) -> None:
    llm = FakeLLM(
        queries={
            "a": "Trajectory good\nScore: 5",
            "b": "Trajectory not good\nScore: 1",
        },
        sequential_responses=True,
    )
    chain = TrajectoryEvalChain.from_llm(llm=llm, agent_tools=[foo])  # type: ignore
    # Test when ref is not provided
    res = chain.evaluate_agent_trajectory(
        input="What is your favorite food?",
        agent_trajectory=intermediate_steps,
        output="I like pie.",
    )
    assert res["score"] == 5
    # Test when ref is provided
    res = chain.evaluate_agent_trajectory(
        input="What is your favorite food?",
        agent_trajectory=intermediate_steps,
        output="I like pie.",
        reference="Paris",
    )
    assert res["score"] == 1


def test_trajectory_eval_chain_no_tools(
    intermediate_steps: List[Tuple[AgentAction, str]]
) -> None:
    llm = FakeLLM(
        queries={
            "a": "Trajectory good\nScore: 5",
            "b": "Trajectory not good\nScore: 1",
        },
        sequential_responses=True,
    )
    chain = TrajectoryEvalChain.from_llm(llm=llm)  # type: ignore
    res = chain.evaluate_agent_trajectory(
        input="What is your favorite food?",
        agent_trajectory=intermediate_steps,
        output="I like pie.",
    )
    assert res["score"] == 5
    res = chain.evaluate_agent_trajectory(
        input="What is your favorite food?",
        agent_trajectory=intermediate_steps,
        output="I like pie.",
        reference="Paris",
    )
    assert res["score"] == 1


def test_old_api_works(intermediate_steps: List[Tuple[AgentAction, str]]) -> None:
    llm = FakeLLM(
        queries={
            "a": "Trajectory good\nScore: 5",
            "b": "Trajectory not good\nScore: 1",
        },
        sequential_responses=True,
    )
    chain = TrajectoryEvalChain.from_llm(llm=llm)  # type: ignore
    res = chain(
        {
            "question": "What is your favorite food?",
            "agent_trajectory": intermediate_steps,
            "answer": "I like pie.",
        }
    )
    assert res["score"] == 5

    res = chain(
        {
            "question": "What is your favorite food?",
            "agent_trajectory": intermediate_steps,
            "answer": "I like pie.",
            "reference": "Paris",
        }
    )
    assert res["score"] == 1

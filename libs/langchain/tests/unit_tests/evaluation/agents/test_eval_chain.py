"""Test agent trajectory evaluation chain."""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.evaluation.agents.trajectory_eval_chain import (
    TrajectoryEval,
    TrajectoryEvalChain,
    TrajectoryOutputParser,
)
from langchain.pydantic_v1 import Field
from langchain.schema import AgentAction, BaseMessage, OutputParserException
from langchain.tools.base import tool
from tests.unit_tests.llms.fake_chat_model import FakeChatModel


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


class _FakeTrajectoryChatModel(FakeChatModel):
    queries: Dict = Field(default_factory=dict)
    sequential_responses: Optional[bool] = False
    response_index: int = 0

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.sequential_responses:
            response = self.queries[list(self.queries.keys())[self.response_index]]
            self.response_index = self.response_index + 1
            return response
        else:
            prompt = messages[0].content
            return self.queries[prompt]


def test_trajectory_output_parser_parse() -> None:
    trajectory_output_parser = TrajectoryOutputParser()
    text = """Judgment: Given the good reasoning in the final answer
but otherwise poor performance, we give the model a score of 2.

Score: 2"""
    got = trajectory_output_parser.parse(text)
    want = TrajectoryEval(
        score=0.25,
        reasoning="""Judgment: Given the good reasoning in the final answer
but otherwise poor performance, we give the model a score of 2.""",
    )

    assert got["score"] == want["score"]
    assert got["reasoning"] == want["reasoning"]

    with pytest.raises(OutputParserException):
        trajectory_output_parser.parse(
            """Judgment: Given the good reasoning in the final answer
but otherwise poor performance, we give the model a score of 2."""
        )

    with pytest.raises(OutputParserException):
        trajectory_output_parser.parse(
            """Judgment: Given the good reasoning in the final answer
but otherwise poor performance, we give the model a score of 2.

Score: 9"""
        )

    with pytest.raises(OutputParserException):
        trajectory_output_parser.parse(
            """Judgment: Given the good reasoning in the final answer
but otherwise poor performance, we give the model a score of 2.

Score: 10"""
        )

    with pytest.raises(OutputParserException):
        trajectory_output_parser.parse(
            """Judgment: Given the good reasoning in the final answer
but otherwise poor performance, we give the model a score of 2.

Score: 0.1"""
        )

    with pytest.raises(OutputParserException):
        trajectory_output_parser.parse(
            """Judgment: Given the good reasoning in the final answer
but otherwise poor performance, we give the model a score of 2.

Score: One"""
        )


def test_trajectory_eval_chain(
    intermediate_steps: List[Tuple[AgentAction, str]]
) -> None:
    llm = _FakeTrajectoryChatModel(
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
        prediction="I like pie.",
    )
    assert res["score"] == 1.0
    # Test when ref is provided
    res = chain.evaluate_agent_trajectory(
        input="What is your favorite food?",
        agent_trajectory=intermediate_steps,
        prediction="I like pie.",
        reference="Paris",
    )
    assert res["score"] == 0.0


def test_trajectory_eval_chain_no_tools(
    intermediate_steps: List[Tuple[AgentAction, str]]
) -> None:
    llm = _FakeTrajectoryChatModel(
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
        prediction="I like pie.",
    )
    assert res["score"] == 1.0
    res = chain.evaluate_agent_trajectory(
        input="What is your favorite food?",
        agent_trajectory=intermediate_steps,
        prediction="I like pie.",
        reference="Paris",
    )
    assert res["score"] == 0.0


def test_old_api_works(intermediate_steps: List[Tuple[AgentAction, str]]) -> None:
    llm = _FakeTrajectoryChatModel(
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
    assert res["score"] == 1.0

    res = chain(
        {
            "question": "What is your favorite food?",
            "agent_trajectory": intermediate_steps,
            "answer": "I like pie.",
            "reference": "Paris",
        }
    )
    assert res["score"] == 0.0

import pytest

from langchain.agents.mrkl.output_parser import (
    MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
    MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
    MRKLOutputParser,
)
from langchain.schema import AgentAction, AgentFinish, OutputParserException

mrkl_output_parser = MRKLOutputParser()


def test_valid_action_and_action_input_parse() -> None:
    llm_output = """I can use the `foo` tool to achieve the goal.
    Action: foo
    Action Input: bar"""

    agent_action: AgentAction = mrkl_output_parser.parse(llm_output)  # type: ignore
    assert agent_action.tool == "foo"
    assert agent_action.tool_input == "bar"


def test_valid_final_answer_parse() -> None:
    llm_output = """Final Answer: The best pizza to eat is margaritta """

    agent_finish: AgentFinish = mrkl_output_parser.parse(llm_output)  # type: ignore
    assert (
        agent_finish.return_values.get("output")
        == "The best pizza to eat is margaritta"
    )


def test_missing_action() -> None:
    llm_output = """I can use the `foo` tool to achieve the goal."""

    with pytest.raises(OutputParserException) as exception_info:
        mrkl_output_parser.parse(llm_output)
    assert (
        exception_info.value.observation == MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE
    )


def test_missing_action_input() -> None:
    llm_output = """I can use the `foo` tool to achieve the goal.
    Action: foo"""

    with pytest.raises(OutputParserException) as exception_info:
        mrkl_output_parser.parse(llm_output)
    assert (
        exception_info.value.observation
        == MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE
    )


def test_final_answer_before_parsable_action() -> None:
    llm_output = """Final Answer: The best pizza to eat is margaritta

        Action: foo
        Action Input: bar
        """
    agent_finish: AgentFinish = mrkl_output_parser.parse(llm_output)  # type: ignore
    assert (
        agent_finish.return_values.get("output")
        == "The best pizza to eat is margaritta"
    )


def test_final_answer_after_parsable_action() -> None:
    llm_output = """
        Observation: I can use the `foo` tool to achieve the goal.
        Action: foo
        Action Input: bar
        Final Answer: The best pizza to eat is margaritta 
        """
    with pytest.raises(OutputParserException) as exception_info:
        mrkl_output_parser.parse(llm_output)
    assert (
        "Parsing LLM output produced both a final answer and a parse-able action"
        in exception_info.value.args[0]
    )

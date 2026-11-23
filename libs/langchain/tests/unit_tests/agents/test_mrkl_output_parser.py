import signal
import sys

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain_classic.agents.mrkl.output_parser import (
    MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
    MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
    MRKLOutputParser,
)

mrkl_output_parser = MRKLOutputParser()


def test_valid_action_and_action_input_parse() -> None:
    llm_output = """I can use the `foo` tool to achieve the goal.
    Action: foo
    Action Input: bar"""

    agent_action: AgentAction = mrkl_output_parser.parse(llm_output)  # type: ignore[assignment]
    assert agent_action.tool == "foo"
    assert agent_action.tool_input == "bar"


def test_valid_final_answer_parse() -> None:
    llm_output = """Final Answer: The best pizza to eat is margaritta """

    agent_finish: AgentFinish = mrkl_output_parser.parse(llm_output)  # type: ignore[assignment]
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
    agent_finish: AgentFinish = mrkl_output_parser.parse(llm_output)  # type: ignore[assignment]
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


def _timeout_handler(_signum: int, _frame: object) -> None:
    msg = "ReDoS: regex took too long"
    raise TimeoutError(msg)


@pytest.mark.skipif(
    sys.platform == "win32", reason="SIGALRM is not available on Windows"
)
def test_mrkl_output_parser_no_redos() -> None:
    """Regression test for ReDoS caused by catastrophic backtracking."""
    malicious = "Action: " + " \t" * 1000 + "Action "
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(2)
    try:
        try:
            mrkl_output_parser.parse(malicious)
        except OutputParserException:
            pass
        except TimeoutError:
            pytest.fail(
                "ReDoS detected: MRKLOutputParser.parse() hung on crafted input"
            )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

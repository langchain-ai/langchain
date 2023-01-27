"""Test that GetMultipleOutputsChain can run successfully."""

from langchain.chains.multiple_outputs.base import GetMultipleOutputsChain
from tests.unit_tests.llms.fake_llm import FakeLLM
from tests.unit_tests.prompts.fake_parser import FakeDictParser


def test_multiple_outputs_can_run() -> None:
    """Test that GetMultipleOutputsChain can run successfully with multiple steps."""
    chain = GetMultipleOutputsChain(
        llm=FakeLLM(
            ensure_and_remove_stop=True,
            sequenced_responses=['fake tool"', 'fake input"'],
        ),
        prefix="Figure out what to do next.\n\n",
        variables={"tool": "Action", "tool_input": "Action Input"},
    )
    assert chain({}) == {
        "tool": "fake tool",
        "tool_input": "fake input",
    }


def test_multiple_outputs_can_run_in_one_step() -> None:
    """Test that GetMultipleOutputsChain can run successfully in a single step."""
    chain = GetMultipleOutputsChain(
        llm=FakeLLM(
            ensure_and_remove_stop=True,
            sequenced_responses=["fake tool\nAction Input: fake input\nObservation:"],
        ),
        prefix="Figure out what to do next.\n\n",
        variables={"tool": "Action", "tool_input": "Action Input"},
        one_step=True,
        one_step_stop="Observation:",
        output_parser=FakeDictParser(),
    )
    assert chain({}) == {
        "tool": "Fake tool",
        "tool_input": "Fake input",
    }

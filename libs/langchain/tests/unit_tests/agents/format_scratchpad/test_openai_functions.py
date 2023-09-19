from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_functions,
)
from langchain.schema.agent import AgentAction
from langchain.schema.messages import AIMessage, FunctionMessage


def test_calls_convert_agent_action_to_messages():
    intermediate_steps = [
        (AgentAction(tool="tool1", tool_input="input1", log="log1"), "observation1"),
        (AgentAction(tool="tool2", tool_input="input2", log="log2"), "observation2"),
        (AgentAction(tool="tool3", tool_input="input3", log="log3"), "observation3"),
    ]
    expected_messages = [
        AIMessage(content="log1"),
        FunctionMessage(name="tool1", content="observation1"),
        AIMessage(content="log2"),
        FunctionMessage(name="tool2", content="observation2"),
        AIMessage(content="log3"),
        FunctionMessage(name="tool3", content="observation3"),
    ]

    assert format_to_openai_functions(intermediate_steps) == expected_messages


def test_handles_empty_input_list():
    intermediate_steps = []
    expected_messages = []

    assert format_to_openai_functions(intermediate_steps) == expected_messages

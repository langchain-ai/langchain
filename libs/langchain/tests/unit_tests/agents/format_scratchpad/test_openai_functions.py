from langchain_core.agents import AgentActionMessageLog
from langchain_core.messages import AIMessage, FunctionMessage

from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)


def test_calls_convert_agent_action_to_messages() -> None:
    data1 = {
        "function_call": {
            "name": "tool1",
            "arguments": "input1",
        }
    }
    message1 = AIMessage(content="", data=data1)
    action1 = AgentActionMessageLog(
        tool="tool1", tool_input="input1", log="log1", message_log=[message1]
    )
    data2 = {
        "function_call": {
            "name": "tool2",
            "arguments": "input2",
        }
    }
    message2 = AIMessage(content="", data=data2)
    action2 = AgentActionMessageLog(
        tool="tool2", tool_input="input2", log="log2", message_log=[message2]
    )

    data3 = {
        "function_call": {
            "name": "tool3",
            "arguments": "input3",
        }
    }
    message3 = AIMessage(content="", data=data3)
    action3 = AgentActionMessageLog(
        tool="tool3", tool_input="input3", log="log3", message_log=[message3]
    )

    intermediate_steps = [
        (action1, "observation1"),
        (action2, "observation2"),
        (action3, "observation3"),
    ]
    expected_messages = [
        message1,
        FunctionMessage(name="tool1", content="observation1"),
        message2,
        FunctionMessage(name="tool2", content="observation2"),
        message3,
        FunctionMessage(name="tool3", content="observation3"),
    ]
    output = format_to_openai_function_messages(intermediate_steps)
    assert output == expected_messages


def test_handles_empty_input_list() -> None:
    output = format_to_openai_function_messages([])
    assert output == []

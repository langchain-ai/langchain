from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain.agents import create_agent
from langchain.tools import tool
from tests.unit_tests.agents.model import FakeToolCallingModel


class InvalidToolCallingModel(FakeToolCallingModel):
    invalid_tool_call_id: str | None = "call_1"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = (messages, stop, run_manager, kwargs)
        if self.index == 0:
            message = AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "get_weather",
                        "args": '{"city":',
                        "id": self.invalid_tool_call_id,
                        "error": "Invalid JSON",
                    }
                ],
            )
        else:
            message = AIMessage(content="Please retry the request.")
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return city


def test_create_agent_answers_invalid_tool_calls() -> None:
    agent = create_agent(InvalidToolCallingModel(), [get_weather])

    result = agent.invoke({"messages": [HumanMessage("Weather?")]})

    assert len(result["messages"]) == 4
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "call_1"
    assert tool_message.name == "get_weather"
    assert tool_message.status == "error"
    assert "malformed or truncated" in tool_message.text
    assert result["messages"][-1].text == "Please retry the request."


def test_create_agent_ignores_invalid_tool_calls_without_ids() -> None:
    model = InvalidToolCallingModel(invalid_tool_call_id=None)
    agent = create_agent(model, [get_weather])

    result = agent.invoke({"messages": [HumanMessage("Weather?")]})

    assert model.index == 1
    assert len(result["messages"]) == 2

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ExtendedModelResponse
from langchain.tools import tool
from tests.unit_tests.agents.model import FakeToolCallingModel


class InjectToolMessageMiddleware(AgentMiddleware):
    """Inject synthetic tool messages for pre-approved tool calls."""

    name = "inject_tool_msg"

    def wrap_model_call(self, request, handler):  # noqa: ANN001
        response = handler(request)
        ai = response.result[0]
        if isinstance(ai, AIMessage) and ai.tool_calls:
            synthetic = [
                ToolMessage(content="cached", tool_call_id=tc["id"], name=tc["name"])
                for tc in ai.tool_calls
            ]
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"messages": synthetic}),
            )
        return response


@tool
def foo(x: int) -> int:
    """A trivial tool the agent can call."""
    return x + 1


def test_create_agent_routes_back_to_model_after_synthetic_tool_messages() -> None:
    """Middleware-injected tool messages should not crash graph routing."""
    fake = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "foo",
                    "args": {"x": 1},
                    "id": "call_xyz",
                    "type": "tool_call",
                }
            ],
            [],
        ]
    )
    agent = create_agent(model=fake, tools=[foo], middleware=[InjectToolMessageMiddleware()])

    result = agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert any(isinstance(message, AIMessage) for message in result["messages"])

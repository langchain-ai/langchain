from typing import Any

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import SystemMessage

from ..model import FakeToolCallingModel


class _AppendMiddleware(AgentMiddleware):
    def __init__(self, label: str, children: list[AgentMiddleware] | None = None) -> None:
        # No tools registered by default
        self.tools = []
        # Optional nested middleware
        self.middleware = children or []
        self._label = label

    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:  # type: ignore[override]
        return {"messages": [SystemMessage(self._label)]}


def test_nested_middleware_ordering() -> None:
    # Build nested chain: X -> Y -> Z
    z = _AppendMiddleware("Z")
    y = _AppendMiddleware("Y", [z])
    x = _AppendMiddleware("X", [y])

    # Siblings before and after X
    a = _AppendMiddleware("A")
    b = _AppendMiddleware("B")

    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        system_prompt=None,
        middleware=[a, x, b],
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
    # The FakeToolCallingModel joins message contents with '-'
    final_ai = result["messages"][-1]
    content: str = final_ai.content  # type: ignore[assignment]

    # Ensure correct in-order appearance: A -> X -> Y -> Z -> B
    assert content.index("A") < content.index("X") < content.index("Y") < content.index("Z") < content.index("B")

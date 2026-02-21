import logging
from collections.abc import Callable
from unittest.mock import MagicMock

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse


class MyMiddleware(AgentMiddleware):
    def __init__(self, id_val: str):
        self.id_val = id_val
        super().__init__()

    @property
    def name(self) -> str:
        return "my_middleware"

    def wrap_model_call(
        self,
        request: ModelRequest,
        forward: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # call the model
        response = forward(request)
        # Append our ID to the content of the first message to prove we ran
        if response.result and isinstance(response.result[0], AIMessage):
            content = response.result[0].content
            response.result[0].content = f"{content} | Middleware: {self.id_val}"
        return response


def test_create_agent_duplicate_middleware_last_wins(caplog):
    """Test that create_agent allows duplicate middleware names but warns and dedups."""
    m1 = MyMiddleware("first")
    m2 = MyMiddleware("second")

    # Mock model
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    # Mock invoke to return a basic message
    model.invoke.return_value = AIMessage(content="Hello")
    model.bind_tools.return_value = model  # simple mock binding
    model.bind.return_value = model

    with caplog.at_level(logging.WARNING):
        # Should not raise AssertionError
        graph = create_agent(model=model, middleware=[m1, m2])

    # Assert warning was logged
    assert "Duplicate middleware names found" in caplog.text

    # Run the graph to see which middleware wraps the call
    # Input state
    result = graph.invoke({"messages": [("user", "hi")]})

    # Check the last message content
    last_msg = result["messages"][-1]

    # We expect "second" to be present, and "first" NOT to be present
    assert "| Middleware: second" in last_msg.content
    assert "first" not in last_msg.content

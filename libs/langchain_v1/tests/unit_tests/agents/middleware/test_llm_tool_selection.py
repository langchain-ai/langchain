"""Unit tests for LLM tool selection middleware."""

import typing
from typing import Union, Any, Literal

from itertools import cycle
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, ModelRequest, modify_model_request
from langchain.agents.middleware import LLMToolSelectorMiddleware
from langchain.messages import AIMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: 72°F, sunny"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    return f"Result of {expression}: 42"


@tool
def send_email(to: str, subject: str) -> str:
    """Send an email to someone."""
    return f"Email sent to {to}"


@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol."""
    return f"Stock price for {symbol}: $150.25"


class FakeModel(GenericFakeChatModel):
    tool_style: Literal["openai", "anthropic"] = "openai"

    def bind_tools(
        self,
        tools: typing.Sequence[Union[dict[str, Any], type[BaseModel], typing.Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if len(tools) == 0:
            msg = "Must provide at least one tool"
            raise ValueError(msg)

        tool_dicts = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_dicts.append(tool)
                continue
            if not isinstance(tool, BaseTool):
                msg = "Only BaseTool and dict is supported by FakeToolCallingModel.bind_tools"
                raise TypeError(msg)

            # NOTE: this is a simplified tool spec for testing purposes only
            if self.tool_style == "openai":
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                        },
                    }
                )
            elif self.tool_style == "anthropic":
                tool_dicts.append(
                    {
                        "name": tool.name,
                    }
                )

        return self.bind(tools=tool_dicts)


class TestLLMToolSelectorBasic:
    """Test basic tool selection functionality."""

    def test_sync_basic_selection(self) -> None:
        """Test synchronous tool selection."""
        # First call: selector picks tools
        # Second call: agent uses selected tools
        tool_calls = [
            [
                {
                    "name": "ToolSelectionResponse",
                    "id": "1",
                    "args": {"tools": ["get_weather", "calculate"]},
                }
            ],
            [{"name": "get_weather", "id": "2", "args": {"location": "Paris"}}],
        ]

        model_requests = []

        @modify_model_request
        def trace_model_requests(request: ModelRequest, state: AgentState, runtime) -> ModelRequest:
            """Middleware to select relevant tools based on state/context."""
            # Select a small, relevant subset of tools based on state/context
            model_requests.append(request)
            return request

        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather", "calculate"]},
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "2", "args": {"location": "Paris"}}
                        ],
                    ),
                    AIMessage(content="The weather in Paris is 72°F and sunny."),
                ]
            )
        )

        tool_selector = LLMToolSelectorMiddleware(max_tools=2, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate, send_email, get_stock_price],
            middleware=[tool_selector, trace_model_requests],
        )

        response = agent.invoke({"messages": [HumanMessage("What's the weather in Paris?")]})

        assert isinstance(response["messages"][-1], AIMessage)

        for request in model_requests:
            selected_tool_names = [tool.name for tool in request.tools] if request.tools else []
            assert selected_tool_names == ["get_weather", "calculate"]

    async def test_async_basic_selection(self) -> None:
        """Test asynchronous tool selection."""
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["search_web"]},
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search_web", "id": "2", "args": {"query": "Python"}}],
                    ),
                    AIMessage(content="Search results found."),
                ]
            )
        )

        tool_selector = LLMToolSelectorMiddleware(max_tools=1, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate],
            middleware=[tool_selector],
        )

        response = await agent.ainvoke({"messages": [HumanMessage("Search for Python tutorials")]})

        assert isinstance(response["messages"][-1], AIMessage)

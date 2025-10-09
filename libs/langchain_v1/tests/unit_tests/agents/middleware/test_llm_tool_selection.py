"""Unit tests for LLM tool selection middleware."""

import typing
from typing import Union, Any, Literal

from itertools import cycle
from pydantic import BaseModel

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, ModelRequest, on_model_call
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

        @on_model_call
        def trace_model_requests(request, handler):
            """Middleware to select relevant tools based on state/context."""
            # Select a small, relevant subset of tools based on state/context
            model_requests.append(request)
            return handler(request)

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


class TestMaxToolsLimiting:
    """Test max_tools limiting behavior."""

    def test_max_tools_limits_selection(self) -> None:
        """Test that max_tools limits selection when model selects too many tools."""
        model_requests = []

        @on_model_call
        def trace_model_requests(request, handler):
            model_requests.append(request)
            return handler(request)

        # Selector model tries to select 4 tools
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {
                                    "tools": [
                                        "get_weather",
                                        "search_web",
                                        "calculate",
                                        "send_email",
                                    ]
                                },
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        # But max_tools=2, so only first 2 should be used
        tool_selector = LLMToolSelectorMiddleware(max_tools=2, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate, send_email],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Verify only 2 tools were passed to the main model
        assert len(model_requests) > 0
        for request in model_requests:
            assert len(request.tools) == 2
            tool_names = [tool.name for tool in request.tools]
            # Should be first 2 from the selection
            assert tool_names == ["get_weather", "search_web"]

    def test_no_max_tools_uses_all_selected(self) -> None:
        """Test that when max_tools is None, all selected tools are used."""
        model_requests = []

        @on_model_call
        def trace_model_requests(request, handler):
            model_requests.append(request)
            return handler(request)

        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {
                                    "tools": [
                                        "get_weather",
                                        "search_web",
                                        "calculate",
                                        "get_stock_price",
                                    ]
                                },
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        # No max_tools specified
        tool_selector = LLMToolSelectorMiddleware(model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate, send_email, get_stock_price],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # All 4 selected tools should be present
        assert len(model_requests) > 0
        for request in model_requests:
            assert len(request.tools) == 4
            tool_names = [tool.name for tool in request.tools]
            assert set(tool_names) == {
                "get_weather",
                "search_web",
                "calculate",
                "get_stock_price",
            }


class TestAlwaysInclude:
    """Test always_include functionality."""

    def test_always_include_tools_present(self) -> None:
        """Test that always_include tools are always present in the request."""
        model_requests = []

        @on_model_call
        def trace_model_requests(request, handler):
            model_requests.append(request)
            return handler(request)

        # Selector picks only search_web
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

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        # But send_email is always included
        tool_selector = LLMToolSelectorMiddleware(
            max_tools=1, always_include=["send_email"], model=tool_selection_model
        )

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, send_email],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Both selected and always_include tools should be present
        assert len(model_requests) > 0
        for request in model_requests:
            tool_names = [tool.name for tool in request.tools]
            assert "search_web" in tool_names
            assert "send_email" in tool_names
            assert len(tool_names) == 2

    def test_always_include_not_counted_against_max(self) -> None:
        """Test that always_include tools don't count against max_tools limit."""
        model_requests = []

        @on_model_call
        def trace_model_requests(request, handler):
            model_requests.append(request)
            return handler(request)

        # Selector picks 2 tools
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather", "search_web"]},
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        # max_tools=2, but we also have 2 always_include tools
        tool_selector = LLMToolSelectorMiddleware(
            max_tools=2,
            always_include=["send_email", "calculate"],
            model=tool_selection_model,
        )

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate, send_email],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Should have 2 selected + 2 always_include = 4 total
        assert len(model_requests) > 0
        for request in model_requests:
            assert len(request.tools) == 4
            tool_names = [tool.name for tool in request.tools]
            assert "get_weather" in tool_names
            assert "search_web" in tool_names
            assert "send_email" in tool_names
            assert "calculate" in tool_names

    def test_multiple_always_include_tools(self) -> None:
        """Test that multiple always_include tools are all present."""
        model_requests = []

        @on_model_call
        def trace_model_requests(request, handler):
            model_requests.append(request)
            return handler(request)

        # Selector picks 1 tool
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {"tools": ["get_weather"]},
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        tool_selector = LLMToolSelectorMiddleware(
            max_tools=1,
            always_include=["send_email", "calculate", "get_stock_price"],
            model=tool_selection_model,
        )

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, send_email, calculate, get_stock_price],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Should have 1 selected + 3 always_include = 4 total
        assert len(model_requests) > 0
        for request in model_requests:
            assert len(request.tools) == 4
            tool_names = [tool.name for tool in request.tools]
            assert "get_weather" in tool_names
            assert "send_email" in tool_names
            assert "calculate" in tool_names
            assert "get_stock_price" in tool_names


class TestDuplicateAndInvalidTools:
    """Test handling of duplicate and invalid tool selections."""

    def test_duplicate_tool_selection_deduplicated(self) -> None:
        """Test that duplicate tool selections are deduplicated."""
        model_requests = []

        @on_model_call
        def trace_model_requests(request, handler):
            model_requests.append(request)
            return handler(request)

        # Selector returns duplicates
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {
                                    "tools": [
                                        "get_weather",
                                        "get_weather",
                                        "search_web",
                                        "search_web",
                                    ]
                                },
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        tool_selector = LLMToolSelectorMiddleware(max_tools=5, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Duplicates should be removed
        assert len(model_requests) > 0
        for request in model_requests:
            tool_names = [tool.name for tool in request.tools]
            assert tool_names == ["get_weather", "search_web"]
            assert len(tool_names) == 2

    def test_max_tools_with_duplicates(self) -> None:
        """Test that max_tools works correctly with duplicate selections."""
        model_requests = []

        @on_model_call
        def trace_model_requests(request, handler):
            model_requests.append(request)
            return handler(request)

        # Selector returns duplicates but max_tools=2
        tool_selection_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ToolSelectionResponse",
                                "id": "1",
                                "args": {
                                    "tools": [
                                        "get_weather",
                                        "get_weather",
                                        "search_web",
                                        "search_web",
                                        "calculate",
                                    ]
                                },
                            }
                        ],
                    ),
                ]
            )
        )

        model = FakeModel(messages=iter([AIMessage(content="Done")]))

        tool_selector = LLMToolSelectorMiddleware(max_tools=2, model=tool_selection_model)

        agent = create_agent(
            model=model,
            tools=[get_weather, search_web, calculate],
            middleware=[tool_selector, trace_model_requests],
        )

        agent.invoke({"messages": [HumanMessage("test")]})

        # Should deduplicate and respect max_tools
        assert len(model_requests) > 0
        for request in model_requests:
            tool_names = [tool.name for tool in request.tools]
            assert len(tool_names) == 2
            assert "get_weather" in tool_names
            assert "search_web" in tool_names

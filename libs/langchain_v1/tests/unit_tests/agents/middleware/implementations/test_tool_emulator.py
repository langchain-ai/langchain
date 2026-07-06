"""Unit tests for tool emulator middleware."""

from collections.abc import Callable, Sequence
from itertools import cycle
from typing import Any, Literal

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain.messages import AIMessage


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    msg = "This tool should be emulated"
    raise NotImplementedError(msg)


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    msg = "This tool should be emulated"
    raise NotImplementedError(msg)


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    # This tool executes normally (not emulated)
    return f"Result: {eval(expression)}"  # noqa: S307


class FakeModel(GenericFakeChatModel):
    """Fake model that supports bind_tools."""

    tool_style: Literal["openai", "anthropic"] = "openai"

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool],
        **_kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        if len(tools) == 0:
            msg = "Must provide at least one tool"
            raise ValueError(msg)

        tool_dicts = []
        for tool_ in tools:
            if isinstance(tool_, dict):
                tool_dicts.append(tool_)
                continue
            if not isinstance(tool_, BaseTool):
                msg = "Only BaseTool and dict is supported by FakeModel.bind_tools"
                raise TypeError(msg)

            # NOTE: this is a simplified tool spec for testing purposes only
            if self.tool_style == "openai":
                tool_dicts.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_.name,
                        },
                    }
                )
            elif self.tool_style == "anthropic":
                tool_dicts.append(
                    {
                        "name": tool_.name,
                    }
                )

        return self.bind(tools=tool_dicts)


class FakeEmulatorModel(BaseChatModel):
    """Fake model for emulating tool responses."""

    responses: Sequence[str] = ("Emulated response",)
    response_index: int = 0

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        response = self.responses[self.response_index % len(self.responses)]
        self.response_index += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "fake_emulator"


class TestLLMToolEmulatorBasic:
    """Test basic tool emulator functionality."""

    def test_emulates_specified_tool_by_name(self) -> None:
        """Test that tools specified by name are emulated."""
        # Model that will call the tool
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "1", "args": {"location": "Paris"}}
                        ],
                    ),
                    AIMessage(content="The weather has been retrieved."),
                ]
            )
        )

        # Model that emulates tool responses
        emulator_model = FakeEmulatorModel(responses=["Emulated: 72°F, sunny in Paris"])

        emulator = LLMToolEmulator(tools=["get_weather"], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather, calculator],
            middleware=[emulator],
        )

        result = agent.invoke({"messages": [HumanMessage("What's the weather in Paris?")]})

        # Should complete without raising NotImplementedError
        assert isinstance(result["messages"][-1], AIMessage)

    def test_emulates_specified_tool_by_instance(self) -> None:
        """Test that tools specified by BaseTool instance are emulated."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search_web", "id": "1", "args": {"query": "Python"}}],
                    ),
                    AIMessage(content="Search results retrieved."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Emulated: Python is a programming language"])

        emulator = LLMToolEmulator(tools=[search_web], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[search_web, calculator],
            middleware=[emulator],
        )

        result = agent.invoke({"messages": [HumanMessage("Search for Python")]})

        assert isinstance(result["messages"][-1], AIMessage)

    def test_non_emulated_tools_execute_normally(self) -> None:
        """Test that tools not in tools_to_emulate execute normally."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "calculator", "id": "1", "args": {"expression": "2+2"}}
                        ],
                    ),
                    AIMessage(content="The calculation is complete."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Should not be used"])

        # Only emulate get_weather, not calculator
        emulator = LLMToolEmulator(tools=["get_weather"], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather, calculator],
            middleware=[emulator],
        )

        result = agent.invoke({"messages": [HumanMessage("Calculate 2+2")]})

        # Calculator should execute normally and return Result: 4
        tool_messages = [
            msg for msg in result["messages"] if hasattr(msg, "name") and msg.name == "calculator"
        ]
        assert len(tool_messages) > 0
        assert "Result: 4" in tool_messages[0].content

    def test_empty_tools_to_emulate_does_nothing(self) -> None:
        """Test that empty tools_to_emulate list means no emulation occurs."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "calculator", "id": "1", "args": {"expression": "5*5"}}
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Should not be used"])

        emulator = LLMToolEmulator(tools=[], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[calculator],
            middleware=[emulator],
        )

        result = agent.invoke({"messages": [HumanMessage("Calculate 5*5")]})

        # Calculator should execute normally
        tool_messages = [
            msg for msg in result["messages"] if hasattr(msg, "name") and msg.name == "calculator"
        ]
        assert len(tool_messages) > 0
        assert "Result: 25" in tool_messages[0].content

    def test_none_tools_emulates_all(self) -> None:
        """Test that None tools means ALL tools are emulated (emulate_all behavior)."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "1", "args": {"location": "NYC"}}
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Emulated: 65°F in NYC"])

        # tools=None means emulate ALL tools
        emulator = LLMToolEmulator(tools=None, model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather],
            middleware=[emulator],
        )

        result = agent.invoke({"messages": [HumanMessage("What's the weather in NYC?")]})

        # Should complete without raising NotImplementedError
        # (get_weather would normally raise NotImplementedError)
        assert isinstance(result["messages"][-1], AIMessage)


class TestLLMToolEmulatorMultipleTools:
    """Test emulating multiple tools."""

    def test_emulate_multiple_tools(self) -> None:
        """Test that multiple tools can be emulated."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "1", "args": {"location": "Paris"}},
                            {"name": "search_web", "id": "2", "args": {"query": "Paris"}},
                        ],
                    ),
                    AIMessage(content="Both tools executed."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(
            responses=["Emulated weather: 20°C", "Emulated search results for Paris"]
        )

        emulator = LLMToolEmulator(tools=["get_weather", "search_web"], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather, search_web, calculator],
            middleware=[emulator],
        )

        result = agent.invoke({"messages": [HumanMessage("Get weather and search for Paris")]})

        # Both tools should be emulated without raising NotImplementedError
        assert isinstance(result["messages"][-1], AIMessage)

    def test_mixed_emulated_and_real_tools(self) -> None:
        """Test that some tools can be emulated while others execute normally."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "1", "args": {"location": "NYC"}},
                            {"name": "calculator", "id": "2", "args": {"expression": "10*2"}},
                        ],
                    ),
                    AIMessage(content="Both completed."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Emulated: 65°F in NYC"])

        # Only emulate get_weather
        emulator = LLMToolEmulator(tools=["get_weather"], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather, calculator],
            middleware=[emulator],
        )

        result = agent.invoke({"messages": [HumanMessage("Weather and calculate")]})

        tool_messages = [msg for msg in result["messages"] if hasattr(msg, "name")]
        assert len(tool_messages) >= 2

        # Calculator should have real result
        calc_messages = [msg for msg in tool_messages if msg.name == "calculator"]
        assert len(calc_messages) > 0
        assert "Result: 20" in calc_messages[0].content


class TestLLMToolEmulatorModelConfiguration:
    """Test custom model configuration for emulation."""

    def test_custom_model_string(self) -> None:
        """Test passing a model string for emulation."""
        # Just test that initialization works - don't require anthropic package
        try:
            emulator = LLMToolEmulator(
                tools=["get_weather"], model="anthropic:claude-sonnet-4-5-20250929"
            )
            assert emulator.model is not None
            assert "get_weather" in emulator.tools_to_emulate
        except ImportError:
            # If anthropic isn't installed, that's fine for this unit test
            pass

    def test_custom_model_instance(self) -> None:
        """Test passing a BaseChatModel instance for emulation."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search_web", "id": "1", "args": {"query": "test"}}],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        custom_emulator_model = FakeEmulatorModel(responses=["Custom emulated response"])

        emulator = LLMToolEmulator(tools=["search_web"], model=custom_emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[search_web],
            middleware=[emulator],
        )

        result = agent.invoke({"messages": [HumanMessage("Search for test")]})

        # Should use the custom model for emulation
        assert isinstance(result["messages"][-1], AIMessage)

    def test_default_model_used_when_none(self) -> None:
        """Test that default model is used when model=None."""
        # Just test that initialization doesn't fail - don't require anthropic package
        # The actual default model requires langchain_anthropic which may not be installed
        try:
            emulator = LLMToolEmulator(tools=["get_weather"], model=None)
            assert emulator.model is not None
        except ImportError:
            # If anthropic isn't installed, that's fine for this unit test
            # The integration tests will verify the full functionality
            pass


class TestLLMToolEmulatorAsync:
    """Test async tool emulator functionality."""

    async def test_async_emulates_specified_tool_by_name(self) -> None:
        """Test that tools specified by name are emulated in async mode."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "1", "args": {"location": "Paris"}}
                        ],
                    ),
                    AIMessage(content="The weather has been retrieved."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Emulated: 72°F, sunny in Paris"])

        emulator = LLMToolEmulator(tools=["get_weather"], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather, calculator],
            middleware=[emulator],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("What's the weather in Paris?")]})

        # Should complete without raising NotImplementedError
        assert isinstance(result["messages"][-1], AIMessage)

    async def test_async_emulates_specified_tool_by_instance(self) -> None:
        """Test that tools specified by BaseTool instance are emulated in async mode."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search_web", "id": "1", "args": {"query": "Python"}}],
                    ),
                    AIMessage(content="Search results retrieved."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Emulated: Python is a programming language"])

        emulator = LLMToolEmulator(tools=[search_web], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[search_web, calculator],
            middleware=[emulator],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("Search for Python")]})

        assert isinstance(result["messages"][-1], AIMessage)

    async def test_async_non_emulated_tools_execute_normally(self) -> None:
        """Test that tools not in tools_to_emulate execute normally in async mode."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "calculator", "id": "1", "args": {"expression": "2+2"}}
                        ],
                    ),
                    AIMessage(content="The calculation is complete."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Should not be used"])

        # Only emulate get_weather, not calculator
        emulator = LLMToolEmulator(tools=["get_weather"], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather, calculator],
            middleware=[emulator],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("Calculate 2+2")]})

        # Calculator should execute normally and return Result: 4
        tool_messages = [
            msg for msg in result["messages"] if hasattr(msg, "name") and msg.name == "calculator"
        ]
        assert len(tool_messages) > 0
        assert "Result: 4" in tool_messages[0].content

    async def test_async_none_tools_emulates_all(self) -> None:
        """Test that None tools means ALL tools are emulated in async mode."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "1", "args": {"location": "NYC"}}
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Emulated: 65°F in NYC"])

        # tools=None means emulate ALL tools
        emulator = LLMToolEmulator(tools=None, model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather],
            middleware=[emulator],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("What's the weather in NYC?")]})

        # Should complete without raising NotImplementedError
        assert isinstance(result["messages"][-1], AIMessage)

    async def test_async_emulate_multiple_tools(self) -> None:
        """Test that multiple tools can be emulated in async mode."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "1", "args": {"location": "Paris"}},
                            {"name": "search_web", "id": "2", "args": {"query": "Paris"}},
                        ],
                    ),
                    AIMessage(content="Both tools executed."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(
            responses=["Emulated weather: 20°C", "Emulated search results for Paris"]
        )

        emulator = LLMToolEmulator(tools=["get_weather", "search_web"], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather, search_web, calculator],
            middleware=[emulator],
        )

        result = await agent.ainvoke(
            {"messages": [HumanMessage("Get weather and search for Paris")]}
        )

        # Both tools should be emulated without raising NotImplementedError
        assert isinstance(result["messages"][-1], AIMessage)

    async def test_async_mixed_emulated_and_real_tools(self) -> None:
        """Test that some tools can be emulated while others execute normally in async mode."""
        agent_model = FakeModel(
            messages=cycle(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "get_weather", "id": "1", "args": {"location": "NYC"}},
                            {"name": "calculator", "id": "2", "args": {"expression": "10*2"}},
                        ],
                    ),
                    AIMessage(content="Both completed."),
                ]
            )
        )

        emulator_model = FakeEmulatorModel(responses=["Emulated: 65°F in NYC"])

        # Only emulate get_weather
        emulator = LLMToolEmulator(tools=["get_weather"], model=emulator_model)

        agent = create_agent(
            model=agent_model,
            tools=[get_weather, calculator],
            middleware=[emulator],
        )

        result = await agent.ainvoke({"messages": [HumanMessage("Weather and calculate")]})

        tool_messages = [msg for msg in result["messages"] if hasattr(msg, "name")]
        assert len(tool_messages) >= 2

        # Calculator should have real result
        calc_messages = [msg for msg in tool_messages if msg.name == "calculator"]
        assert len(calc_messages) > 0
        assert "Result: 20" in calc_messages[0].content

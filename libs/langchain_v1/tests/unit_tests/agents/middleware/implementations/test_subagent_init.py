"""Unit tests for SubAgentMiddleware initialization and configuration."""

import json
from typing import Any, get_type_hints

from langchain.agents.middleware.subagents import SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY, SubAgent, SubAgentMiddleware
import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import AutoStrategy
from langchain.tools import ToolRuntime
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from tests.unit_tests.chat_model import GenericFakeChatModel


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": "General-purpose agent for answering weather questions.",
    "system_prompt": "Use the available tools to answer the user's weather question.",
}


class _DynamicStructuredOutputModel(GenericFakeChatModel):
    """Fake model that calls whichever structured-output tool is bound."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_name = getattr(self.tools[-1], "name", None)
        if not isinstance(tool_name, str):
            msg = "Expected a structured-output tool to be bound"
            raise TypeError(msg)
        self.messages = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": {
                                "name": "Maya Thornton",
                                "age": 29,
                                "city": "Portland",
                            },
                            "id": "call_payload",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


class TestSubagentMiddlewareInit:
    """Tests for SubAgentMiddleware initialization that don't require LLM invocation."""

    @pytest.fixture(autouse=True)
    def set_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set dummy API key for model initialization."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def test_subagent_middleware_init(self) -> None:
        """Test basic SubAgentMiddleware initialization with general-purpose subagent."""
        middleware = SubAgentMiddleware(
            subagents=[
                {
                    **GENERAL_PURPOSE_SUBAGENT,
                    "model": "gpt-5.4-mini",
                    "tools": [],
                }
            ],
        )
        assert middleware is not None
        assert "Available subagent types:" in middleware.system_prompt
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"

    def test_task_tool_compiles_dynamic_response_format_for_declarative_subagent(self) -> None:
        """Dynamic schemas are present when declarative subagent variants compile."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
        }
        response_format = AutoStrategy(schema)
        middleware = SubAgentMiddleware(
            subagents=[
                {
                    "name": "worker",
                    "description": "Does work.",
                    "system_prompt": "Return structured data.",
                    "model": _DynamicStructuredOutputModel(messages=iter(())),
                    "tools": [],
                }
            ],
            system_prompt=None,
        )
        task_tool = middleware.tools[0]
        runtime = ToolRuntime(
            state={},
            context={},
            config={
                "configurable": {
                    SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY: response_format,
                }
            },
            stream_writer=lambda _chunk: None,
            tools=[task_tool],
            tool_call_id="call_worker",
            store=None,
        )

        result = task_tool.func(
            description="Make a person.",
            subagent_type="worker",
            runtime=runtime,
        )

        assert json.loads(result.update["messages"][0].content) == {
            "name": "Maya Thornton",
            "age": 29,
            "city": "Portland",
        }

    def test_task_tool_rejects_response_format_for_compiled_subagent(self) -> None:
        """Dynamic schemas require a declarative spec to compile a variant."""
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
        response_format = AutoStrategy(schema)

        runnable = RunnableLambda(lambda _state, _config: (_ for _ in ()).throw(AssertionError("compiled runnable should not be invoked")))

        task_tool = _build_task_tool(
            [
                {
                    "name": "worker",
                    "description": "Does work.",
                    "runnable": runnable,
                }
            ]
        )
        runtime = ToolRuntime(
            state={},
            context={},
            config={
                "configurable": {
                    SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY: response_format,
                }
            },
            stream_writer=lambda _chunk: None,
            tools=[task_tool],
            tool_call_id="call_worker",
            store=None,
        )

        with pytest.raises(
            ValueError,
            match='response_schema cannot be used with compiled subagent "worker"',
        ):
            task_tool.func(
                description="Do work.",
                subagent_type="worker",
                runtime=runtime,
            )

    def test_subagent_middleware_with_custom_subagent(self) -> None:
        """Test SubAgentMiddleware initialization with a custom subagent."""
        middleware = SubAgentMiddleware(
            subagents=[
                {
                    "name": "weather",
                    "description": "Weather subagent",
                    "system_prompt": "Get weather.",
                    "model": "gpt-5.4-mini",
                    "tools": [get_weather],
                }
            ],
        )
        assert middleware is not None
        # System prompt includes TASK_SYSTEM_PROMPT plus available subagent types
        assert middleware.system_prompt.startswith(TASK_SYSTEM_PROMPT)
        assert "weather" in middleware.system_prompt

    def test_subagent_middleware_custom_system_prompt(self) -> None:
        """Test SubAgentMiddleware with a custom system prompt."""
        middleware = SubAgentMiddleware(
            subagents=[
                {
                    "name": "weather",
                    "description": "Weather subagent",
                    "system_prompt": "Get weather.",
                    "model": "gpt-5.4-mini",
                    "tools": [],
                }
            ],
            system_prompt="Use the task tool to call a subagent.",
        )
        assert middleware is not None
        # Custom system prompt plus available subagent types
        assert middleware.system_prompt.startswith("Use the task tool to call a subagent.")

    def test_requires_subagents(self) -> None:
        """Test that at least one subagent is required."""
        with pytest.raises(ValueError, match="At least one subagent"):
            SubAgentMiddleware(
                subagents=[],
            )

    def test_subagent_requires_model(self) -> None:
        """Test that subagents must specify model."""
        with pytest.raises(ValueError, match="must specify 'model'"):
            SubAgentMiddleware(
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "tools": [],
                        # Missing "model"
                    }
                ],
            )

    def test_subagent_requires_tools(self) -> None:
        """Test that subagents must specify tools."""
        with pytest.raises(ValueError, match="must specify 'tools'"):
            SubAgentMiddleware(
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "model": "gpt-5.4-mini",
                        # Missing "tools"
                    }
                ],
            )

    def _make_echo_graph(self) -> object:
        """Build a minimal MessagesState graph for use in CompiledSubAgent tests."""

        def echo_node(_state: MessagesState) -> dict:
            return {"messages": [AIMessage(content="hello")]}

        builder = StateGraph(MessagesState)
        builder.add_node("echo", echo_node)
        builder.add_edge(START, "echo")
        return builder.compile()

    def _task_runtime(self, task_tool: object, tool_call_id: str) -> ToolRuntime:
        return ToolRuntime(
            state={},
            context={},
            config={"configurable": {}},
            stream_writer=lambda _chunk: None,
            tools=[task_tool],
            tool_call_id=tool_call_id,
            store=None,
        )

    def test_compiled_subagent_name_propagated_via_config(self) -> None:
        """CompiledSubAgent.name is forwarded into metadata.lc_agent_name and run_name."""
        configs: list[dict[str, object]] = []

        class _Runnable:
            def with_config(self, config: dict[str, object]) -> "_Runnable":
                configs.append(config)
                return self

            def invoke(
                self,
                state: dict[str, object],
                config: object = None,
            ) -> dict[str, object]:
                del state, config
                return {"messages": [AIMessage(content="hello")]}

        SubAgentMiddleware(
            subagents=[
                {
                    "name": "my-subagent",
                    "description": "A custom subagent",
                    "runnable": _Runnable(),
                }
            ],
        )

        assert configs == [
            {
                "metadata": {"lc_agent_name": "my-subagent"},
                "run_name": "my-subagent",
            }
        ]

    def test_compiled_subagent_does_not_mutate_original_runnable(self) -> None:
        """Task-tool setup must not mutate the original runnable."""
        graph = self._make_echo_graph()
        original_config = getattr(graph, "config", None)

        SubAgentMiddleware(
            subagents=[
                {
                    "name": "my-subagent",
                    "description": "A custom subagent",
                    "runnable": graph,
                }
            ],
        )

        assert graph.config == original_config, "Original runnable was mutated; use with_config instead of attribute assignment"

    def test_same_runnable_reused_across_multiple_subagents(self) -> None:
        """Same runnable registered under two different names must not cross-contaminate configs."""

        class _Runnable:
            def __init__(self, config: dict[str, object] | None = None) -> None:
                self.config = config

            def with_config(self, config: dict[str, object]) -> "_Runnable":
                return _Runnable(config)

            def invoke(
                self,
                state: dict[str, object],
                config: object = None,
            ) -> dict[str, object]:
                del state, config
                if self.config is None:
                    msg = "Expected configured runnable clone"
                    raise AssertionError(msg)
                return {"messages": [AIMessage(content=str(self.config["run_name"]))]}

        graph = _Runnable()

        middleware = SubAgentMiddleware(
            subagents=[
                {
                    "name": "agent-alpha",
                    "description": "First binding",
                    "runnable": graph,
                },
                {
                    "name": "agent-beta",
                    "description": "Second binding",
                    "runnable": graph,
                },
            ],
        )

        task_tool = middleware.tools[0]
        alpha = task_tool.func(
            description="Do alpha work.",
            subagent_type="agent-alpha",
            runtime=self._task_runtime(task_tool, "call_alpha"),
        )
        beta = task_tool.func(
            description="Do beta work.",
            subagent_type="agent-beta",
            runtime=self._task_runtime(task_tool, "call_beta"),
        )

        assert alpha.update["messages"][0].content == "agent-alpha"
        assert beta.update["messages"][0].content == "agent-beta"
        assert graph.config is None

    def test_middleware_delegates_to_create_sub_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Middleware should use the shared entrypoint for declarative subagents."""
        graph = self._make_echo_graph()
        calls: list[tuple[object, type | None]] = []

        class CustomState(MessagesState):
            pass

        def fake_create_sub_agent(
            spec: object,
            *,
            state_schema: type | None = None,
            response_format: object = None,
        ) -> object:
            del response_format
            calls.append((spec, state_schema))
            return graph

        monkeypatch.setattr("deepagents.middleware.subagents.create_sub_agent", fake_create_sub_agent)

        SubAgentMiddleware(
            subagents=[
                {
                    "name": "agent-alpha",
                    "description": "First binding",
                    "system_prompt": "Work on the task.",
                    "model": "test-model",
                    "tools": [],
                },
            ],
            state_schema=CustomState,
        )

        assert len(calls) == 1
        assert calls[0][1] is CustomState

    def test_multiple_subagents_with_interrupt_on(self) -> None:
        """Test creating agent with multiple subagents that have interrupt_on configured."""
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the task tool to call subagents.",
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        {
                            "name": "subagent1",
                            "description": "First subagent.",
                            "system_prompt": "You are subagent 1.",
                            "model": "claude-sonnet-4-6",
                            "tools": [get_weather],
                        },
                        {
                            "name": "subagent2",
                            "description": "Second subagent.",
                            "system_prompt": "You are subagent 2.",
                            "model": "claude-sonnet-4-6",
                            "tools": [get_weather],
                        },
                    ],
                )
            ],
        )
        # This would error if the middleware was accumulated incorrectly
        assert agent is not None

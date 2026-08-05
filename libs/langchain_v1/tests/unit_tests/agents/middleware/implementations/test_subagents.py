"""Tests for sub-agent middleware functionality.

This module contains tests for the subagent system, focusing on how subagents
are invoked, how they return results, and how state is managed between parent
and child agents.
"""

import dataclasses
import json
import re
import uuid
from collections.abc import Callable, Iterator, Sequence
from typing import Annotated, Any, TypedDict, cast

from typing_extensions import override
from unittest.mock import MagicMock

from langchain.agents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState, PrivateStateAttr
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from langsmith import Client
from langsmith.run_helpers import tracing_context
from pydantic import BaseModel, Field



class GenericFakeChatModel(BaseChatModel):
    """Local fake chat model for subagent tests."""

    messages: Iterator[AIMessage | str] = Field(exclude=True)
    call_history: list[Any] = Field(default_factory=list)
    tools: Sequence[dict[str, Any] | type | Callable | BaseTool] = ()
    stream_delimiter: str | None = None

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tools = tools
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_history.append(
            {
                "messages": messages,
                "kwargs": {"stop": stop, "run_manager": run_manager, **kwargs},
                "tools": self.tools,
            }
        )
        message = next(self.messages)
        message_ = AIMessage(content=message) if isinstance(message, str) else message
        return ChatResult(generations=[ChatGeneration(message=message_)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs).generations[0].message
        if not isinstance(message, AIMessage):
            msg = f"Expected invoke to return an AIMessage, got {type(message)}."
            raise ValueError(msg)

        content = message.content
        tool_calls = message.tool_calls
        if content:
            if not isinstance(content, str):
                msg = "Expected content to be a string."
                raise ValueError(msg)
            chunks = [content] if self.stream_delimiter is None else [chunk for chunk in cast("list[str]", re.split(self.stream_delimiter, content)) if chunk]
            for idx, token in enumerate(chunks):
                is_last = idx == len(chunks) - 1
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=token,
                        id=message.id,
                        tool_calls=tool_calls if is_last else [],
                    )
                )
                if is_last and not message.additional_kwargs:
                    chunk.message.chunk_position = "last"
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        elif tool_calls:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="", id=message.id, tool_calls=tool_calls, chunk_position="last"
                )
            )
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk

        for key, value in message.additional_kwargs.items():
            if key == "function_call":
                for fkey, fvalue in value.items():
                    value_chunks = (
                        cast("list[str]", re.split(r"(,)", fvalue))
                        if isinstance(fvalue, str)
                        else [fvalue]
                    )
                    for value_chunk in value_chunks:
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(
                                id=message.id,
                                content="",
                                additional_kwargs={"function_call": {fkey: value_chunk}},
                            )
                        )
                        if run_manager:
                            run_manager.on_llm_new_token("", chunk=chunk)
                        yield chunk
            else:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        id=message.id, content="", additional_kwargs={key: value}
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token("", chunk=chunk)
                yield chunk

    @property
    def _llm_type(self) -> str:
        return "generic-fake-chat-model"


class _ScriptedChatModel(BaseChatModel):
    """Fake chat model that returns a fixed scripted sequence of AIMessages.

    Each call to `_generate` returns the next message in `responses`;
    once exhausted, it repeats the final response. This avoids `StopIteration`
    bugs that arise with plain iterators under langgraph's generator runner.
    """

    responses: list[AIMessage] = []  # noqa: RUF012  # Pydantic field, per-instance
    tools: Sequence[dict[str, Any] | type | Callable | BaseTool] = ()
    _call_idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = min(self._call_idx, len(self.responses) - 1)
        self._call_idx += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tools = tools
        return self


class TestSubAgents:
    """Tests for sub-agent middleware functionality."""

    def test_subagent_returns_final_message_as_tool_result(self) -> None:
        """Test that a subagent's final message is returned as a ToolMessage.

        This test verifies the core subagent functionality:
        1. Parent agent invokes the 'task' tool to launch a subagent
        2. Subagent executes and returns a result
        3. The subagent's final message is extracted and returned to the parent
           as a ToolMessage in the parent's message list
        4. Only the final message content is included (not the full conversation)

        The response flow is:
        - Parent receives ToolMessage with content from subagent's last AIMessage
        - State updates (excluding messages/todos/structured_response) are merged
        - Parent can then process the subagent's response and continue
        """
        # Create the parent agent's chat model that will call the subagent
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: invoke the task tool to launch subagent
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Calculate the sum of 2 and 3",
                                    "subagent_type": "general-purpose",
                                },
                                "id": "call_calculate_sum",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Second response: acknowledge the subagent's result
                    AIMessage(content="The calculation has been completed."),
                ]
            )
        )

        # Create the subagent's chat model that will handle the calculation
        subagent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="The sum of 2 and 3 is 5."),
                ]
            )
        )

        # Create the compiled subagent
        compiled_subagent = create_agent(model=subagent_chat_model)

        # Create the parent agent with subagent support
        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        CompiledSubAgent(
                            name="general-purpose",
                            description="A general-purpose agent for various tasks.",
                            runnable=compiled_subagent,
                        ),
                    ]
                )
            ]
        )

        # Invoke the parent agent with an initial message
        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="What is 2 + 3?")]},
            config={"configurable": {"thread_id": "test_thread_calculation"}},
        )

        # Verify the result contains messages
        assert "messages" in result, "Result should contain messages key"
        assert len(result["messages"]) > 0, "Result should have at least one message"

        # Find the ToolMessage that contains the subagent's response
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0, "Should have at least one ToolMessage from subagent"

        # Verify the ToolMessage contains the subagent's final response
        subagent_tool_message = tool_messages[0]
        assert "The sum of 2 and 3 is 5." in subagent_tool_message.content, "ToolMessage should contain subagent's final message content"

    def test_multiple_subagents_invoked_in_parallel(self) -> None:
        """Test that multiple different subagents can be launched in parallel.

        This test verifies parallel execution with distinct subagent types:
        1. Parent agent makes a single AIMessage with multiple tool_calls
        2. Two different subagents are invoked concurrently (math-adder and math-multiplier)
        3. Each specialized subagent completes its task independently
        4. Both subagent results are returned as separate ToolMessages
        5. Parent agent receives both results and can synthesize them

        The parallel execution pattern is important for:
        - Reducing latency when tasks are independent
        - Efficient resource utilization
        - Handling multi-part user requests with specialized agents
        """
        # Create the parent agent's chat model that will call both subagents in parallel
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: invoke TWO different task tools in parallel
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Calculate the sum of 5 and 7",
                                    "subagent_type": "math-adder",
                                },
                                "id": "call_addition",
                                "type": "tool_call",
                            },
                            {
                                "name": "task",
                                "args": {
                                    "description": "Calculate the product of 4 and 6",
                                    "subagent_type": "math-multiplier",
                                },
                                "id": "call_multiplication",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    # Second response: acknowledge both results
                    AIMessage(content="Both calculations completed successfully."),
                ]
            )
        )

        # Create specialized subagent models - each handles a specific math operation
        addition_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="The sum of 5 and 7 is 12."),
                ]
            )
        )

        multiplication_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="The product of 4 and 6 is 24."),
                ]
            )
        )

        # Compile the two different specialized subagents
        addition_subagent = create_agent(model=addition_subagent_model)
        multiplication_subagent = create_agent(model=multiplication_subagent_model)

        # Create the parent agent with BOTH specialized subagents
        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        CompiledSubAgent(
                            name="math-adder",
                            description="Specialized agent for addition operations.",
                            runnable=addition_subagent,
                        ),
                        CompiledSubAgent(
                            name="math-multiplier",
                            description="Specialized agent for multiplication operations.",
                            runnable=multiplication_subagent,
                        ),
                    ]
                )
            ],
        )

        # Invoke the parent agent with a request that triggers parallel subagent calls
        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="What is 5+7 and what is 4*6?")]},
            config={"configurable": {"thread_id": "test_thread_parallel"}},
        )

        # Verify the result contains messages
        assert "messages" in result, "Result should contain messages key"

        # Find all ToolMessages - should have one for each subagent invocation
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2, f"Should have exactly 2 ToolMessages (one per subagent), but got {len(tool_messages)}"

        # Create a lookup map from tool_call_id to ToolMessage for precise verification
        tool_messages_by_id = {msg.tool_call_id: msg for msg in tool_messages}

        # Verify we have both expected tool call IDs
        assert "call_addition" in tool_messages_by_id, "Should have response from addition subagent"
        assert "call_multiplication" in tool_messages_by_id, "Should have response from multiplication subagent"

        # Verify the exact content of each response by looking up the specific tool message
        addition_tool_message = tool_messages_by_id["call_addition"]
        assert addition_tool_message.content == "The sum of 5 and 7 is 12.", (
            f"Addition subagent should return exact message, got: {addition_tool_message.content}"
        )

        multiplication_tool_message = tool_messages_by_id["call_multiplication"]
        assert multiplication_tool_message.content == "The product of 4 and 6 is 24.", (
            f"Multiplication subagent should return exact message, got: {multiplication_tool_message.content}"
        )

    def test_private_state_does_not_propagate_between_sibling_subagents(self) -> None:
        """A private state field should not propagate from one sibling subagent to another."""

        class _LocalPrivateState(AgentState):
            shared_value: Annotated[str | None, PrivateStateAttr]

        class _LocalPrivateMiddleware(AgentMiddleware[_LocalPrivateState, Any, Any]):
            state_schema = _LocalPrivateState

            def before_agent(self, state: _LocalPrivateState, runtime: object) -> dict[str, Any] | None:
                if "shared_value" in state:
                    return None
                return {"shared_value": "seeded"}

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Seed the interpreter state",
                                    "subagent_type": "writer",
                                },
                                "id": "call_writer",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Read the interpreter state",
                                    "subagent_type": "reader",
                                },
                                "id": "call_reader",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )

        writer_model = GenericFakeChatModel(messages=iter([AIMessage(content="writer saw seeded")]))
        reader_model = GenericFakeChatModel(messages=iter([AIMessage(content="reader saw missing")]))

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        SubAgent(
                            name="writer",
                            description="Writes state.",
                            system_prompt="Write the seeded private state value and report completion.",
                            model=writer_model,
                            middleware=[_LocalPrivateMiddleware()],
                        ),
                        SubAgent(
                            name="reader",
                            description="Reads state.",
                            system_prompt="Read the private state value and report what you received.",
                            model=reader_model,
                        ),
                    ],
                )
            ]
        )

        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="run the two subagents")]},
            config={"configurable": {"thread_id": "test_shared_quickjs_subagents"}},
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2
        assert "seeded" in tool_messages[0].content
        assert "missing" in tool_messages[1].content

    def test_private_state_does_not_propagate_from_parent_to_subagent(self) -> None:
        """A private state field on the parent should not be visible to a child subagent."""

        class _ParentPrivateState(AgentState):
            shared_value: Annotated[str | None, PrivateStateAttr]

        class _ChildCaptureState(AgentState):
            shared_value: Annotated[str | None, PrivateStateAttr]

        captured_child_states: list[dict[str, Any]] = []

        class _ChildCaptureMiddleware(AgentMiddleware[_ChildCaptureState, Any, Any]):
            state_schema = _ChildCaptureState

            def before_agent(self, state: _ChildCaptureState, runtime: object) -> dict[str, Any] | None:
                captured_child_states.append(dict(state))
                return None

        class _ParentSeedMiddleware(AgentMiddleware[_ParentPrivateState, Any, Any]):
            state_schema = _ParentPrivateState

            def before_agent(self, state: _ParentPrivateState, runtime: object) -> dict[str, Any] | None:
                if "shared_value" in state:
                    return None
                return {"shared_value": "parent-secret"}

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Run the child subagent",
                                    "subagent_type": "child",
                                },
                                "id": "call_child",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )

        child_model = GenericFakeChatModel(messages=iter([AIMessage(content="child done")]))

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[_ParentSeedMiddleware(),
                SubAgentMiddleware(
                    subagents=[
                        SubAgent(
                            name="child",
                            description="Captures its incoming state.",
                            system_prompt="Capture the incoming state and complete the task.",
                            model=child_model,
                            middleware=[_ChildCaptureMiddleware()],
                        ),
                    ]
                )
            ],
        )

        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="run the child subagent")]},
            config={
                "configurable": {"thread_id": "test_private_state_parent_to_child"},
                "metadata": {"shared_value": "parent-secret"},
            },
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "child done" in tool_messages[0].content
        assert captured_child_states, "Child subagent should have received state"
        assert "shared_value" not in captured_child_states[0]

    def test_private_state_from_custom_state_schema_does_not_propagate_to_subagent(self) -> None:
        """Private fields on `create_agent(state_schema=...)` should not reach subagents."""

        class _ParentState(AgentState):
            parent_secret: Annotated[str | None, PrivateStateAttr]
            public_value: str | None

        captured_subagent_states: list[dict[str, Any]] = []

        @tool
        def seed_parent_state(runtime: ToolRuntime) -> Command:
            """Seed parent state."""
            return Command(
                update={
                    "parent_secret": "parent-secret",
                    "public_value": "parent-public",
                    "messages": [ToolMessage(content="seeded", tool_call_id=runtime.tool_call_id)],
                }
            )

        @tool
        def capture_subagent_state(query: str, runtime: ToolRuntime) -> str:
            """Capture subagent state."""
            captured_subagent_states.append(dict(runtime.state))
            return f"captured {query}"

        parent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "seed_parent_state",
                                "args": {},
                                "id": "call_seed_parent_state",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Capture the incoming state",
                                    "subagent_type": "child",
                                },
                                "id": "call_child",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )
        child_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_subagent_state",
                                "args": {"query": "state"},
                                "id": "call_capture_subagent_state",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="child done"),
                ]
            )
        )

        parent_agent = create_agent(
            model=parent_model,
            tools=[seed_parent_state],
            checkpointer=InMemorySaver(),
            state_schema=_ParentState,
            middleware=[
                SubAgentMiddleware(
                    state_schema=_ParentState,
                    subagents=[
                        SubAgent(
                            name="child",
                            description="Captures its incoming state.",
                            system_prompt="Capture the incoming state and complete the task.",
                            model=child_model,
                            tools=[capture_subagent_state],
                        )
                    ],
                )
            ]
        )

        parent_agent.invoke(
            {"messages": [HumanMessage(content="seed state and run the child subagent")]},
            config={"configurable": {"thread_id": "test_private_state_schema_parent_to_child"}},
        )

        assert captured_subagent_states, "Child subagent should have captured state"
        assert captured_subagent_states[0]["public_value"] == "parent-public"
        assert "parent_secret" not in captured_subagent_states[0]

    def test_agent_with_structured_output_tool_strategy(self) -> None:
        """Test that an agent with ToolStrategy properly generates structured output.

        This test verifies the structured output setup:
        1. Define a Pydantic model as the response schema
        2. Configure agent with ToolStrategy for structured output
        3. Fake model calls the structured output tool
        4. Agent validates and returns the structured response
        5. The structured_response key contains the validated Pydantic instance

        This validates our understanding of how to set up structured output
        correctly using the fake model for testing.
        """

        # Define the Pydantic model for structured output
        class WeatherReport(BaseModel):
            """Structured weather information."""

            location: str = Field(description="The city or location for the weather report")
            temperature: float = Field(description="Temperature in Celsius")
            condition: str = Field(description="Weather condition (e.g., sunny, rainy)")

        # Create a fake model that calls the structured output tool
        # The tool name will be the schema class name: "WeatherReport"
        fake_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "WeatherReport",
                                "args": {
                                    "location": "San Francisco",
                                    "temperature": 18.5,
                                    "condition": "sunny",
                                },
                                "id": "call_weather_report",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            )
        )

        # Create agent with ToolStrategy for structured output
        agent = create_agent(
            model=fake_model,
            response_format=ToolStrategy(schema=WeatherReport),
        )

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content="What's the weather in San Francisco?")]})

        # Verify the structured_response key exists in the result
        assert "structured_response" in result, "Result should contain structured_response key"

        # Verify the structured response is the correct type
        structured_response = result["structured_response"]
        assert isinstance(structured_response, WeatherReport), f"Expected WeatherReport instance, got {type(structured_response)}"

        # Verify the structured response has the correct values
        expected_response = WeatherReport(location="San Francisco", temperature=18.5, condition="sunny")
        assert structured_response == expected_response, f"Expected {expected_response}, got {structured_response}"

    def test_parallel_subagents_with_todo_lists(self) -> None:
        """Test that multiple subagents can manage their own isolated todo lists.

        This test verifies that:
        1. Multiple subagents can be invoked in parallel
        2. Each subagent can use write_todos to manage its own todo list
        3. Todo lists are properly isolated to each subagent (not merged into parent)
        4. Parent receives clean ToolMessages from each subagent
        5. The 'todos' key is excluded from parent state per _EXCLUDED_STATE_KEYS

        This validates that todo list state isolation works correctly in parallel execution.
        """
        # Create parent agent's chat model that calls two subagents in parallel
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: invoke TWO subagents in parallel
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Research the history of Python programming language",
                                    "subagent_type": "python-researcher",
                                },
                                "id": "call_research_python",
                                "type": "tool_call",
                            },
                            {
                                "name": "task",
                                "args": {
                                    "description": "Research the history of JavaScript programming language",
                                    "subagent_type": "javascript-researcher",
                                },
                                "id": "call_research_javascript",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    # Second response: acknowledge both results
                    AIMessage(content="Both research tasks completed successfully."),
                ]
            )
        )

        # Create first subagent that uses write_todos and returns a result
        python_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First: write some todos
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "Search for Python history",
                                            "status": "in_progress",
                                            "activeForm": "Searching for Python history",
                                        },
                                        {"content": "Summarize findings", "status": "pending", "activeForm": "Summarizing findings"},
                                    ]
                                },
                                "id": "call_write_todos_python_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Second: update todos and return final message
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {"content": "Search for Python history", "status": "completed", "activeForm": "Searching for Python history"},
                                        {"content": "Summarize findings", "status": "completed", "activeForm": "Summarizing findings"},
                                    ]
                                },
                                "id": "call_write_todos_python_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Final result message
                    AIMessage(content="Python was created by Guido van Rossum and released in 1991."),
                ]
            )
        )

        # Create second subagent that uses write_todos and returns a result
        javascript_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First: write some todos
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "Search for JavaScript history",
                                            "status": "in_progress",
                                            "activeForm": "Searching for JavaScript history",
                                        },
                                        {"content": "Compile summary", "status": "pending", "activeForm": "Compiling summary"},
                                    ]
                                },
                                "id": "call_write_todos_js_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Second: update todos and return final message
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {
                                    "todos": [
                                        {
                                            "content": "Search for JavaScript history",
                                            "status": "completed",
                                            "activeForm": "Searching for JavaScript history",
                                        },
                                        {"content": "Compile summary", "status": "completed", "activeForm": "Compiling summary"},
                                    ]
                                },
                                "id": "call_write_todos_js_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Final result message
                    AIMessage(content="JavaScript was created by Brendan Eich at Netscape in 1995."),
                ]
            )
        )

        python_research_agent = create_agent(
            model=python_subagent_model,
            middleware=[TodoListMiddleware()],
        )

        javascript_research_agent = create_agent(
            model=javascript_subagent_model,
            middleware=[TodoListMiddleware()],
        )

        # Create parent agent with both specialized subagents
        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="python-researcher",
                    description="Agent specialized in Python research.",
                    runnable=python_research_agent,
                ),
                CompiledSubAgent(
                    name="javascript-researcher",
                    description="Agent specialized in JavaScript research.",
                    runnable=javascript_research_agent,
                ),
            ])],
        )

        # Invoke the parent agent
        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="Research Python and JavaScript history")]},
            config={"configurable": {"thread_id": "test_thread_todos"}},
        )

        # Verify the result contains messages
        assert "messages" in result, "Result should contain messages key"

        # Find all ToolMessages from the subagents
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2, f"Should have exactly 2 ToolMessages, got {len(tool_messages)}"

        # Create lookup map by tool_call_id
        tool_messages_by_id = {msg.tool_call_id: msg for msg in tool_messages}

        # Verify both expected tool call IDs are present
        assert "call_research_python" in tool_messages_by_id, "Should have response from Python researcher"
        assert "call_research_javascript" in tool_messages_by_id, "Should have response from JavaScript researcher"

        # Verify that todos are NOT in the parent agent's final state
        # (they should be excluded per _EXCLUDED_STATE_KEYS)
        assert "todos" not in result, "Parent agent state should not contain todos key (it should be excluded per _EXCLUDED_STATE_KEYS)"

        # Verify the final messages contain the research results
        python_tool_message = tool_messages_by_id["call_research_python"]
        assert "Python was created by Guido van Rossum" in python_tool_message.content, (
            f"Expected Python research result in message, got: {python_tool_message.content}"
        )

        javascript_tool_message = tool_messages_by_id["call_research_javascript"]
        assert "JavaScript was created by Brendan Eich" in javascript_tool_message.content, (
            f"Expected JavaScript research result in message, got: {javascript_tool_message.content}"
        )

    def test_subagent_propagates_recursion_limit_to_tool_runtime(self) -> None:
        """Test that subagent tools receive the parent's recursion limit via `ToolRuntime.config`."""
        captured_config: Any = None

        @tool
        def capture_recursion_limit(runtime: ToolRuntime) -> str:
            """Capture the recursion limit from runtime config."""
            nonlocal captured_config
            captured_config = runtime.config
            return "OK"

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Check the recursion limit and report it.",
                                    "subagent_type": "general-purpose",
                                },
                                "id": "call_subagent_recursion_limit",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="The subagent finished successfully."),
                ]
            )
        )

        subagent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_recursion_limit",
                                "args": {},
                                "id": "call_capture_recursion_limit",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )

        compiled_subagent = create_agent(
            model=subagent_chat_model,
            tools=[capture_recursion_limit],
            name="subagent-runtime-check",
        ).with_config({"recursion_limit": 5000})

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="general-purpose",
                    description="A general-purpose agent for various tasks.",
                    runnable=compiled_subagent,
                )
            ])],
        )

        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="Run the recursion limit check.")]},
            config={
                "configurable": {"thread_id": str(uuid.uuid4())},
                "tags": ["hello"],
            },
            durability="exit",
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert captured_config is not None
        assert captured_config["recursion_limit"] == 5000
        # Pregel merges the runtime recursion_limit patch with the subagent's own
        # config instead of replacing it wholesale.
        assert captured_config["tags"] == ["hello"]
        # CompiledSubAgent.name takes precedence over the name set in create_agent()
        # so that lc_agent_name in streamed chunks reflects the declared subagent name.
        assert captured_config["metadata"]["lc_agent_name"] == "general-purpose"

    def test_subagent_inherits_parent_user_metadata(self) -> None:
        """User metadata set on the parent invoke reaches subagent runs (deepagents#3634).

        `langgraph`'s `ensure_config` seeds each run's metadata from the ambient
        parent config and merges it per-key (langgraph#7926). A user key like
        `customer_id` therefore propagates into subagent runs, while the
        subagent's bound `lc_agent_name` wins the key collision and is preserved.

        Requires a `langgraph` that includes langgraph#7926's merge semantics;
        with the older overwrite behaviour the parent metadata is dropped.
        """
        captured_config: Any = None

        @tool
        def capture_metadata(runtime: ToolRuntime) -> str:
            """Capture the runtime config from inside the subagent."""
            nonlocal captured_config
            captured_config = runtime.config
            return "OK"

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Capture metadata and report it.",
                                    "subagent_type": "general-purpose",
                                },
                                "id": "call_subagent_metadata",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="The subagent finished successfully."),
                ]
            )
        )

        subagent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_metadata",
                                "args": {},
                                "id": "call_capture_metadata",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )

        compiled_subagent = create_agent(
            model=subagent_chat_model,
            tools=[capture_metadata],
            name="subagent-runtime-check",
        )

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="general-purpose",
                    description="A general-purpose agent for various tasks.",
                    runnable=compiled_subagent,
                )
            ])],
        )

        parent_agent.invoke(
            {"messages": [HumanMessage(content="Run the metadata check.")]},
            config={
                "configurable": {"thread_id": str(uuid.uuid4())},
                # `lc_agent_name` collides with the subagent's bound identity (it
                # must keep its own value); `customer_id` is a non-colliding user
                # key that must survive the merge into the subagent's runs.
                "metadata": {"customer_id": "abc-123", "lc_agent_name": "parent-agent"},
            },
            durability="exit",
        )

        assert captured_config is not None
        subagent_metadata = captured_config["metadata"]
        # User-set parent metadata propagated into the subagent run.
        assert subagent_metadata["customer_id"] == "abc-123"
        # The subagent's bound identity won the `lc_agent_name` collision.
        assert subagent_metadata["lc_agent_name"] == "general-purpose"

    @pytest.mark.xfail(
        reason="callbacks in parent config are not forwarded to subagent invocations (see #2315)",
        strict=True,
    )
    def test_subagent_propagates_callbacks_to_model_calls(self) -> None:
        """Test that callbacks in parent config are forwarded to subagent model invocations.

        Regression test for https://github.com/langchain-ai/deepagents/issues/2315.
        """
        llm_start_agent_names: list[str] = []

        class CapturingCallback(BaseCallbackHandler):
            def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
                llm_start_agent_names.append(kwargs.get("name", "unknown"))

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Do something.",
                                    "subagent_type": "general-purpose",
                                },
                                "id": "call_subagent_callback",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        subagent_chat_model = GenericFakeChatModel(messages=iter([AIMessage(content="Subagent done.")]))

        compiled_subagent = create_agent(
            model=subagent_chat_model,
            name="callback-check-subagent",
        )

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="general-purpose",
                    description="A general-purpose agent.",
                    runnable=compiled_subagent,
                )
            ])],
        )

        callback = CapturingCallback()

        parent_agent.invoke(
            {"messages": [HumanMessage(content="Run the callback check.")]},
            config={
                "configurable": {"thread_id": str(uuid.uuid4())},
                "callbacks": [callback],
            },
            durability="exit",
        )

        # All three LLM calls (2 parent + 1 subagent) should trigger the callback
        assert len(llm_start_agent_names) == 3, (
            f"Expected callbacks from 2 parent + 1 subagent LLM calls, but only got {len(llm_start_agent_names)}: {llm_start_agent_names}"
        )
        # The subagent name should be identifiable in at least one callback
        assert any(name == "callback-check-subagent" for name in llm_start_agent_names), (
            f"Subagent LLM call should have triggered callback with correct name, got: {llm_start_agent_names}"
        )

    def test_parallel_subagents_with_different_structured_outputs(self) -> None:
        """Test that multiple subagents with different structured outputs work correctly.

        This test verifies that:
        1. Two different subagents can be invoked in parallel
        2. Each subagent has its own structured output schema
        3. Structured responses are properly excluded from parent state (per _EXCLUDED_STATE_KEYS)
        4. Parent receives clean ToolMessages from each subagent
        5. Each subagent's structured_response stays isolated to that subagent

        This validates that structured_response exclusion prevents schema conflicts
        between parent and subagent agents.
        """

        # Define structured output schemas for the two specialized subagents
        class CityWeather(BaseModel):
            """Weather information for a city."""

            city: str = Field(description="Name of the city")
            temperature_celsius: float = Field(description="Temperature in Celsius")
            humidity_percent: int = Field(description="Humidity percentage")

        class CityPopulation(BaseModel):
            """Population statistics for a city."""

            city: str = Field(description="Name of the city")
            population: int = Field(description="Total population")
            metro_area_population: int = Field(description="Metropolitan area population")

        # Create parent agent's chat model that calls both subagents in parallel
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    # First response: invoke TWO different subagents in parallel
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Get weather information for Tokyo",
                                    "subagent_type": "weather-analyzer",
                                },
                                "id": "call_weather",
                                "type": "tool_call",
                            },
                            {
                                "name": "task",
                                "args": {
                                    "description": "Get population statistics for Tokyo",
                                    "subagent_type": "population-analyzer",
                                },
                                "id": "call_population",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    # Second response: acknowledge both results
                    AIMessage(content="I've gathered weather and population data for Tokyo."),
                ]
            )
        )

        # Create weather subagent with structured output
        weather_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "CityWeather",
                                "args": {
                                    "city": "Tokyo",
                                    "temperature_celsius": 22.5,
                                    "humidity_percent": 65,
                                },
                                "id": "call_weather_struct",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            )
        )

        weather_subagent = create_agent(
            model=weather_subagent_model,
            response_format=ToolStrategy(schema=CityWeather),
        )

        # Create population subagent with structured output
        population_subagent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "CityPopulation",
                                "args": {
                                    "city": "Tokyo",
                                    "population": 14000000,
                                    "metro_area_population": 37400000,
                                },
                                "id": "call_population_struct",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            )
        )

        population_subagent = create_agent(
            model=population_subagent_model,
            response_format=ToolStrategy(schema=CityPopulation),
        )

        # Create parent agent with both specialized subagents
        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="weather-analyzer",
                    description="Specialized agent for weather analysis.",
                    runnable=weather_subagent,
                ),
                CompiledSubAgent(
                    name="population-analyzer",
                    description="Specialized agent for population analysis.",
                    runnable=population_subagent,
                ),
            ])],
        )

        # Invoke the parent agent
        result = parent_agent.invoke(
            {"messages": [HumanMessage(content="Tell me about Tokyo's weather and population")]},
            config={"configurable": {"thread_id": "test_thread_structured"}},
        )

        # Verify the result contains messages
        assert "messages" in result, "Result should contain messages key"

        # Find all ToolMessages from the subagents
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2, f"Should have exactly 2 ToolMessages, got {len(tool_messages)}"

        # Create lookup map by tool_call_id
        tool_messages_by_id = {msg.tool_call_id: msg for msg in tool_messages}

        # Verify both expected tool call IDs are present
        assert "call_weather" in tool_messages_by_id, "Should have response from weather subagent"
        assert "call_population" in tool_messages_by_id, "Should have response from population subagent"

        # Verify that structured_response is NOT in the parent agent's final state
        # (it should be excluded per _EXCLUDED_STATE_KEYS)
        assert "structured_response" not in result, (
            "Parent agent state should not contain structured_response key (it should be excluded per _EXCLUDED_STATE_KEYS)"
        )

        # When a subagent produces a structured_response, the ToolMessage content is
        # the JSON-serialized structured data (not the last message text).
        weather_tool_message = tool_messages_by_id["call_weather"]
        weather_parsed = CityWeather.model_validate_json(weather_tool_message.content)
        assert weather_parsed == CityWeather(city="Tokyo", temperature_celsius=22.5, humidity_percent=65), (
            f"Expected JSON-serialized weather data, got: {weather_tool_message.content}"
        )

        population_tool_message = tool_messages_by_id["call_population"]
        population_parsed = CityPopulation.model_validate_json(population_tool_message.content)
        assert population_parsed == CityPopulation(city="Tokyo", population=14000000, metro_area_population=37400000), (
            f"Expected JSON-serialized population data, got: {population_tool_message.content}"
        )

    def test_structured_response_serialized_as_tool_message(self) -> None:
        """Test that structured_response is JSON-serialized as ToolMessage content.

        When a subagent produces a `structured_response`, the middleware should
        JSON-serialize it as the ToolMessage content instead of extracting the
        last message text.
        """
        structured_data = {
            "findings": "Renewable energy adoption is accelerating",
            "confidence": 0.92,
            "sources": 3,
        }

        mock_subagent = RunnableLambda(
            lambda _: {
                "messages": [AIMessage(content="Here are my findings about renewable energy.")],
                "structured_response": structured_data,
            }
        )

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Analyze renewable energy trends",
                                    "subagent_type": "analyzer",
                                },
                                "id": "call_structured",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done"),
                ]
            )
        )

        agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="analyzer",
                    description="An analysis agent",
                    runnable=mock_subagent,
                ),
            ])],
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="Analyze renewable energy")]},
            config={"configurable": {"thread_id": f"test-structured-{uuid.uuid4().hex}"}},
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        task_tool_message = tool_messages[0]
        assert task_tool_message.content == json.dumps(structured_data)

        parsed = json.loads(task_tool_message.content)
        assert parsed == structured_data

    def test_structured_response_dataclass_serialized_as_tool_message(self) -> None:
        """Test that a dataclass structured_response is JSON-serialized correctly.

        Dataclass instances don't have `model_dump_json` and aren't natively
        JSON-serializable, so the middleware must convert them via
        `dataclasses.asdict` before calling `json.dumps`.
        """

        @dataclasses.dataclass
        class AnalysisResult:
            findings: str
            confidence: float
            sources: int

        structured_instance = AnalysisResult(
            findings="Renewable energy adoption is accelerating",
            confidence=0.92,
            sources=3,
        )

        mock_subagent = RunnableLambda(
            lambda _: {
                "messages": [AIMessage(content="Here are my findings.")],
                "structured_response": structured_instance,
            }
        )

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Analyze trends",
                                    "subagent_type": "analyzer",
                                },
                                "id": "call_dc",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done"),
                ]
            )
        )

        agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="analyzer",
                    description="An analysis agent",
                    runnable=mock_subagent,
                ),
            ])],
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="Analyze")]},
            config={"configurable": {"thread_id": f"test-dc-structured-{uuid.uuid4().hex}"}},
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        task_tool_message = tool_messages[0]

        parsed = json.loads(task_tool_message.content)
        assert parsed == {
            "findings": "Renewable energy adoption is accelerating",
            "confidence": 0.92,
            "sources": 3,
        }

    def test_structured_response_pydantic_serialized_as_tool_message(self) -> None:
        """Test that a Pydantic model structured_response uses model_dump_json."""

        class AnalysisResult(BaseModel):
            findings: str
            confidence: float

        structured_instance = AnalysisResult(
            findings="Solar is growing fast",
            confidence=0.95,
        )

        mock_subagent = RunnableLambda(
            lambda _: {
                "messages": [AIMessage(content="Here are my findings.")],
                "structured_response": structured_instance,
            }
        )

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Analyze trends",
                                    "subagent_type": "analyzer",
                                },
                                "id": "call_pydantic",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done"),
                ]
            )
        )

        agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="analyzer",
                    description="An analysis agent",
                    runnable=mock_subagent,
                ),
            ])],
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="Analyze")]},
            config={"configurable": {"thread_id": f"test-pydantic-structured-{uuid.uuid4().hex}"}},
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        task_tool_message = tool_messages[0]

        parsed = AnalysisResult.model_validate_json(task_tool_message.content)
        assert parsed == AnalysisResult(findings="Solar is growing fast", confidence=0.95)

    def test_fallback_to_last_message_without_structured_response(self) -> None:
        """Test fallback to last message when no structured_response is present.

        When a subagent does not produce a `structured_response`, the middleware
        should fall back to extracting the last message text.
        """
        mock_subagent = RunnableLambda(
            lambda _: {
                "messages": [AIMessage(content="Plain text result without structured response")],
            }
        )

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Do work",
                                    "subagent_type": "worker",
                                },
                                "id": "call_plain",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done"),
                ]
            )
        )

        agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="worker",
                    description="A worker agent",
                    runnable=mock_subagent,
                ),
            ])],
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="Test")]},
            config={"configurable": {"thread_id": f"test-no-structured-{uuid.uuid4().hex}"}},
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        task_tool_message = tool_messages[0]
        assert task_tool_message.content == "Plain text result without structured response"

    def test_fallback_skips_trailing_empty_ai_message(self) -> None:
        """Skip a trailing empty AIMessage and use the last AIMessage with text.

        Anthropic/Bedrock occasionally emits an empty `end_turn` AIMessage after
        a successful final tool call. The middleware should walk back to the
        prior AIMessage carrying the real answer instead of forwarding an empty
        ToolMessage.
        """
        mock_subagent = RunnableLambda(
            lambda _: {
                "messages": [
                    AIMessage(content="The real answer from the subagent."),
                    AIMessage(content=""),
                ],
            }
        )

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Do work",
                                    "subagent_type": "worker",
                                },
                                "id": "call_trailing_empty",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done"),
                ]
            )
        )

        agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="worker",
                    description="A worker agent",
                    runnable=mock_subagent,
                ),
            ])],
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="Test")]},
            config={"configurable": {"thread_id": f"test-trailing-empty-{uuid.uuid4().hex}"}},
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "The real answer from the subagent."

    def test_subagent_streaming_emits_messages_and_updates_from_subgraph(self) -> None:
        """Test end-to-end subagent streaming with `subgraphs=True`.

        Verifies:
        1. Parent and subagent message chunks are both streamed in `messages` mode.
        2. Parent and subagent completed messages are both streamed in `updates` mode.
        3. Subagent message metadata includes its `lc_agent_name`, inherited tags, and config metadata.
        4. The subagent's tool result is surfaced back through the parent tools update.
        """
        parent_content = "PARENT_RESPONSE"
        subagent_content = "SUBAGENT_RESPONSE"
        test_tags = ["test-tag", "session-123"]

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "Do task", "subagent_type": "worker"},
                                "id": "call_worker",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content=parent_content),
                ]
            ),
            stream_delimiter="_",
        )
        subagent_chat_model = GenericFakeChatModel(
            messages=iter([AIMessage(content=subagent_content)]),
            stream_delimiter="_",
        )

        compiled_subagent = create_agent(model=subagent_chat_model, name="worker")
        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            name="supervisor",
            middleware=[SubAgentMiddleware(subagents=[CompiledSubAgent(name="worker", description="Does work.", runnable=compiled_subagent)])],
        )

        saw_parent_message_chunk = False
        saw_subagent_message_chunk = False
        saw_subagent_update = False
        saw_parent_tools_update = False
        saw_parent_model_update = False

        seen_agent_names: set[str | None] = set()

        for ns, stream_mode, data in parent_agent.stream(
            {"messages": [HumanMessage(content="Do something")]},
            stream_mode=["messages", "updates"],
            subgraphs=True,
            config={"configurable": {"thread_id": "test_thread"}, "tags": test_tags},
        ):
            if stream_mode == "messages":
                message_chunk, metadata = data
                agent_name = metadata.get("lc_agent_name")
                seen_agent_names.add(agent_name)
                tags = metadata.get("tags", [])

                if parent_content.split("_", maxsplit=1)[0] in message_chunk.content and agent_name == "supervisor":
                    saw_parent_message_chunk = True

                if subagent_content.split("_", maxsplit=1)[0] in message_chunk.content and agent_name == "worker":
                    assert all(t in tags for t in test_tags), f"Subagent chunk missing tags. Expected {test_tags}, got {tags}"
                    saw_subagent_message_chunk = True

            elif stream_mode == "updates":
                update = data
                if "model" in update and ns and ns[-1].startswith("tools:"):
                    subagent_message = update["model"]["messages"][-1]
                    assert subagent_message.content == subagent_content.replace("_", "")
                    saw_subagent_update = True
                elif "tools" in update and ns == ():
                    tool_message = update["tools"]["messages"][-1]
                    assert tool_message.content == subagent_content.replace("_", "")
                    saw_parent_tools_update = True
                elif "model" in update and ns == ():
                    parent_message = update["model"]["messages"][-1]
                    if parent_message.content == parent_content.replace("_", ""):
                        saw_parent_model_update = True

        assert saw_parent_message_chunk, "Should have seen parent message chunks in the stream"
        assert saw_subagent_message_chunk, "Should have seen subagent message chunks in the stream"
        assert saw_subagent_update, "Should have seen a subagent model update in the stream"
        assert saw_parent_tools_update, "Should have seen the parent tools update with the subagent result"
        assert saw_parent_model_update, "Should have seen the parent final model update in the stream"
        assert seen_agent_names == {"supervisor", "worker"}

    def test_compiled_subagent_lc_agent_name_in_stream_metadata(self) -> None:
        """lc_agent_name in streamed chunks must reflect the CompiledSubAgent's declared name.

        Regression test for #2925: when a raw StateGraph (not created via create_agent)
        is passed as a CompiledSubAgent, streamed chunks must carry the declared name in
        metadata, not the parent agent's name.
        """
        subagent_content = "RAW_GRAPH_RESPONSE"

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "Do task", "subagent_type": "raw-worker"},
                                "id": "call_raw_worker",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            ),
            stream_delimiter="_",
        )
        subagent_chat_model = GenericFakeChatModel(
            messages=iter([AIMessage(content=subagent_content)]),
            stream_delimiter="_",
        )

        # Raw StateGraph — NOT created via create_agent, so no lc_agent_name pre-set.
        builder = StateGraph(MessagesState)
        builder.add_node("model", create_agent(model=subagent_chat_model))
        builder.add_edge(START, "model")
        raw_graph = builder.compile()

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            name="supervisor",
            middleware=[SubAgentMiddleware(subagents=[CompiledSubAgent(name="raw-worker", description="Raw graph subagent.", runnable=raw_graph)])],
        )

        seen_subagent_names: set[str | None] = set()

        for _ns, stream_mode, data in parent_agent.stream(
            {"messages": [HumanMessage(content="Do something")]},
            stream_mode=["messages"],
            subgraphs=True,
            config={"configurable": {"thread_id": "test_raw_graph_lc_agent_name"}},
        ):
            if stream_mode == "messages":
                message_chunk, metadata = data
                if message_chunk.content:
                    seen_subagent_names.add(metadata.get("lc_agent_name"))

        assert "raw-worker" in seen_subagent_names, f"Expected 'raw-worker' in streamed lc_agent_name metadata, got: {seen_subagent_names}"

    async def test_compiled_subagent_lc_agent_name_in_astream_metadata(self) -> None:
        """Async variant of the #2925 streaming regression test.

        The fix relies on `with_config` being symmetric across sync/async, but the
        symptom in #2925 also shows up in `astream` — covering both paths guards
        against an async-only regression.
        """
        subagent_content = "RAW_GRAPH_RESPONSE"

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "Do task", "subagent_type": "raw-worker"},
                                "id": "call_raw_worker_async",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            ),
            stream_delimiter="_",
        )
        subagent_chat_model = GenericFakeChatModel(
            messages=iter([AIMessage(content=subagent_content)]),
            stream_delimiter="_",
        )

        builder = StateGraph(MessagesState)
        builder.add_node("model", create_agent(model=subagent_chat_model))
        builder.add_edge(START, "model")
        raw_graph = builder.compile()

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            name="supervisor",
            middleware=[SubAgentMiddleware(subagents=[CompiledSubAgent(name="raw-worker", description="Raw graph subagent.", runnable=raw_graph)])],
        )

        seen_subagent_names: set[str | None] = set()

        async for _ns, stream_mode, data in parent_agent.astream(
            {"messages": [HumanMessage(content="Do something")]},
            stream_mode=["messages"],
            subgraphs=True,
            config={"configurable": {"thread_id": "test_raw_graph_lc_agent_name_async"}},
        ):
            if stream_mode == "messages":
                message_chunk, metadata = data
                if message_chunk.content:
                    seen_subagent_names.add(metadata.get("lc_agent_name"))

        assert "raw-worker" in seen_subagent_names, f"Expected 'raw-worker' in async streamed lc_agent_name metadata, got: {seen_subagent_names}"

    def test_compiled_subagent_name_overrides_inner_runnable_name_in_stream(self) -> None:
        """CompiledSubAgent.name takes precedence over the inner runnable's lc_agent_name in streamed chunks.

        When the inner runnable was itself created via `create_agent(name=...)`, the
        registered `CompiledSubAgent.name` is what the parent uses to reference the
        subagent and what tracing consumers display. This precedence is verified at
        the tool-runtime layer in `test_subagent_propagates_recursion_limit_to_tool_runtime`;
        this test pins the same precedence in the streamed-chunk metadata surface
        from #2925 so a future "fix" that swaps merge order can't silently regress it.
        """
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "Do task", "subagent_type": "outer-name"},
                                "id": "call_named_inner",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            ),
            stream_delimiter="_",
        )
        subagent_chat_model = GenericFakeChatModel(
            messages=iter([AIMessage(content="NAMED_INNER_RESPONSE")]),
            stream_delimiter="_",
        )

        named_inner = create_agent(model=subagent_chat_model, name="inner-name")

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            name="supervisor",
            middleware=[SubAgentMiddleware(subagents=[CompiledSubAgent(name="outer-name", description="Subagent with a different inner name.", runnable=named_inner)])],
        )

        seen_subagent_names: set[str | None] = set()

        for _ns, stream_mode, data in parent_agent.stream(
            {"messages": [HumanMessage(content="Do something")]},
            stream_mode=["messages"],
            subgraphs=True,
            config={"configurable": {"thread_id": "test_outer_name_wins"}},
        ):
            if stream_mode == "messages":
                message_chunk, metadata = data
                if message_chunk.content:
                    seen_subagent_names.add(metadata.get("lc_agent_name"))

        assert "outer-name" in seen_subagent_names, f"Expected 'outer-name' in streamed lc_agent_name metadata, got: {seen_subagent_names}"
        assert "inner-name" not in seen_subagent_names, f"Inner runnable's lc_agent_name leaked into stream metadata: {seen_subagent_names}"

    def test_config_passed_to_runnable_lambda_subagent(self) -> None:
        """Test that config (including tags) is passed to a RunnableLambda subagent.

        RunnableLambda doesn't have a 'config' attribute, so this tests the safe getattr fallback.
        """
        received_configs: list[RunnableConfig] = []

        def lambda_subagent(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:  # noqa: ARG001
            received_configs.append(config)
            return {"messages": [AIMessage(content="Lambda response")]}

        runnable_lambda = RunnableLambda(lambda_subagent)
        assert not hasattr(runnable_lambda, "config")

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "Do something", "subagent_type": "lambda-agent"},
                                "id": "call_lambda",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            name="parent",
            middleware=[SubAgentMiddleware(subagents=[CompiledSubAgent(name="lambda-agent", description="Lambda subagent.", runnable=runnable_lambda)])],
        )

        test_tags = ["lambda-tag", "config-test"]
        parent_agent.invoke(
            {"messages": [HumanMessage(content="Do something")]},
            config={"configurable": {"thread_id": "test_lambda"}, "tags": test_tags},
        )

        assert len(received_configs) > 0, "Lambda should have been invoked"
        assert all(t in received_configs[0].get("tags", []) for t in test_tags), f"Missing tags in config: {received_configs[0].get('tags')}"

    @pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
    def test_context_passed_to_subagent_tool_runtime(self) -> None:
        """Test that context passed to main agent is available in subagent's ToolRuntime.context."""
        received_contexts: list[Any] = []

        @tool
        def capture_context(query: str, runtime: ToolRuntime) -> str:
            """Captures runtime context."""
            received_contexts.append(runtime.context)
            return f"Processed: {query}"

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "Use capture_context", "subagent_type": "ctx-agent"},
                                "id": "call_ctx",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )
        subagent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "capture_context",
                                "args": {"query": "test"},
                                "id": "call_tool",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Captured."),
                ]
            )
        )

        compiled_subagent = create_agent(model=subagent_chat_model, tools=[capture_context], name="ctx-agent")
        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            name="orchestrator",
            middleware=[SubAgentMiddleware(subagents=[CompiledSubAgent(name="ctx-agent", description="Context-aware subagent.", runnable=compiled_subagent)])],
        )

        test_context = {"user_id": "user-123", "session_id": "session-456"}
        parent_agent.invoke(
            {"messages": [HumanMessage(content="Process")]},
            config={"configurable": {"thread_id": "test_context"}},
            context=test_context,
        )

        assert len(received_contexts) > 0, "Subagent tool should have been invoked"
        assert received_contexts[0] == test_context, f"Expected {test_context}, got {received_contexts[0]}"

    def test_compiled_subagent_without_messages_raises_error(self) -> None:
        """Test that a CompiledSubAgent without 'messages' in state raises a clear error.

        This test verifies that when a custom StateGraph is used with CompiledSubAgent
        and doesn't include a 'messages' key in its state, a helpful ValueError is raised
        explaining the requirement.
        """

        # Define a custom state without 'messages' key
        class CustomState(TypedDict):
            custom_field: str

        def custom_node(_state: CustomState) -> CustomState:
            return {"custom_field": "processed"}

        # Build a custom graph that doesn't use messages
        graph_builder = StateGraph(CustomState)
        graph_builder.add_node("process", custom_node)
        graph_builder.add_edge(START, "process")
        graph_builder.add_edge("process", END)
        custom_graph = graph_builder.compile()

        # Create parent agent with this custom subagent
        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Process something",
                                    "subagent_type": "custom-processor",
                                },
                                "id": "call_custom",
                            }
                        ],
                    ),
                ]
            )
        )

        parent_agent = create_agent(
            model=parent_chat_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                CompiledSubAgent(
                    name="custom-processor",
                    description="A custom processor",
                    runnable=custom_graph,
                )
            ])],
        )

        # Attempting to invoke should raise a clear error about missing 'messages' key
        with pytest.raises(
            ValueError,
            match="CompiledSubAgent must return a dict containing a 'messages' key",
        ):
            parent_agent.invoke(
                {"messages": [HumanMessage(content="Process this")]},
                config={"configurable": {"thread_id": "test_thread_no_messages"}},
            )

    def test_ls_agent_type_is_trace_only_metadata(self) -> None:
        """`ls_agent_type` must reach LangSmith but not streamed callback metadata.

        The task tool wraps each subagent invocation in a langsmith
        `tracing_context` with `metadata={"ls_agent_type": "subagent"}` so
        downstream LangSmith tracing can distinguish subagent runs from
        root-agent runs. Because this metadata is set via langsmith's tracing
        contextvar (not via RunnableConfig), it only reaches the
        `LangChainTracer` — it is not added to the callback manager's
        metadata and therefore does not leak into streamed callback events.
        """
        # Root model: first call emits a task tool call that dispatches to the
        # subagent; subsequent calls return a final response.
        root_model = _ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "Do some work",
                                "subagent_type": "test-worker",
                            },
                            "id": "call_test_worker",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
        subagent_model = _ScriptedChatModel(responses=[AIMessage(content="Subagent completed the task.")])

        # Capture streamed callback metadata per-run.
        captured_callbacks: list[dict[str, Any]] = []

        class CaptureHandler(BaseCallbackHandler):
            def on_chain_start(
                self,
                serialized: dict[str, Any],
                inputs: dict[str, Any],
                *,
                run_id: str,
                parent_run_id: str | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                captured_callbacks.append(
                    {
                        "name": kwargs.get("name") or (serialized or {}).get("name"),
                        "metadata": metadata or {},
                    }
                )

        mock_session = MagicMock()
        mock_client = Client(session=mock_session, api_key="test", auto_batch_tracing=False)

        agent = create_agent(
            model=root_model,
            checkpointer=InMemorySaver(),
            middleware=[SubAgentMiddleware(subagents=[
                SubAgent(
                    name="test-worker",
                    description="A test worker subagent.",
                    system_prompt="You are a test worker.",
                    model=subagent_model,
                )
            ])],
        )

        with tracing_context(client=mock_client, enabled=True):
            agent.invoke(
                {"messages": [HumanMessage(content="Please do some work")]},
                config={
                    "configurable": {"thread_id": "test_ls_agent_type"},
                    "callbacks": [CaptureHandler()],
                },
            )

        # The subagent path must have run (otherwise the trace-only check below
        # is trivially true).
        subagent_callback = next((c for c in captured_callbacks if c["name"] == "test-worker"), None)
        assert subagent_callback is not None, f"Expected a 'test-worker' subagent callback, got names: {[c['name'] for c in captured_callbacks]}"

        # (1) ls_agent_type must not leak into any streamed callback metadata.
        for captured in captured_callbacks:
            assert "ls_agent_type" not in captured["metadata"], (
                f"ls_agent_type leaked into callback metadata for run {captured['name']!r}: {captured['metadata']}"
            )

        # (2) ls_agent_type='subagent' must reach the LangSmith tracer.
        posts: list[dict[str, Any]] = []
        for call in mock_session.request.mock_calls:
            if call.args and call.args[0] == "POST":
                body = json.loads(call.kwargs["data"])
                posts.extend(body.get("post", []) if "post" in body else [body])

        subagent_tracer_metadatas = [
            post.get("extra", {}).get("metadata", {})
            for post in posts
            if post.get("extra", {}).get("metadata", {}).get("ls_agent_type") == "subagent"
        ]
        assert subagent_tracer_metadatas, (
            f"Expected at least one LangSmith post with ls_agent_type='subagent'. "
            f"Got tracer metadatas: "
            f"{[p.get('extra', {}).get('metadata', {}) for p in posts]}"
        )

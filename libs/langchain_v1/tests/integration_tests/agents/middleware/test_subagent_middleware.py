import json
from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.subagents import SubAgent, SubAgentMiddleware
from langchain.agents.structured_output import ToolStrategy


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


class WeatherMiddleware(AgentMiddleware):
    tools: ClassVar = [get_weather]


GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": "General-purpose agent for answering weather questions.",
    "system_prompt": "Use the available tools to answer the user's weather question.",
}


def assert_expected_subgraph_actions(expected_tool_calls, agent, inputs):
    current_idx = 0
    for update in agent.stream(
        inputs,
        subgraphs=True,
        stream_mode="updates",
    ):
        if "model" in update[1]:
            ai_message = update[1]["model"]["messages"][-1]
            tool_calls = ai_message.tool_calls
            for tool_call in tool_calls:
                if tool_call["name"] == expected_tool_calls[current_idx]["name"]:
                    if "model" in expected_tool_calls[current_idx]:
                        # Providers may return date-suffixed names
                        expected_model = expected_tool_calls[current_idx]["model"]
                        actual_model = ai_message.response_metadata["model_name"]
                        assert actual_model == expected_model or actual_model.startswith(expected_model + "-"), (
                            f"Expected model {expected_model!r}, got {actual_model!r}"
                        )
                    for arg in expected_tool_calls[current_idx]["args"]:
                        assert arg in tool_call["args"]
                        assert tool_call["args"][arg] == expected_tool_calls[current_idx]["args"][arg]
                    current_idx += 1
    assert current_idx == len(expected_tool_calls)


@pytest.mark.requires("langchain_anthropic", "langchain_openai")
class TestSubagentMiddleware:
    """Integration tests for the SubagentMiddleware class."""

    def test_general_purpose_subagent(self):
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the general-purpose subagent to get the weather in a city.",
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        {
                            **GENERAL_PURPOSE_SUBAGENT,
                            "model": "claude-sonnet-4-6",
                            "tools": [get_weather],
                        }
                    ],
                )
            ],
        )
        assert "task" in agent.nodes["tools"].bound._tools_by_name
        response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
        assert response["messages"][1].tool_calls[0]["name"] == "task"
        assert response["messages"][1].tool_calls[0]["args"]["subagent_type"] == "general-purpose"

    def test_defined_subagent_tool_calls(self):
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "system_prompt": "Use the get_weather tool to get the weather in a city.",
                            "model": "claude-sonnet-4-6",
                            "tools": [get_weather],
                        }
                    ],
                )
            ],
        )
        expected_tool_calls = [
            {"name": "task", "args": {"subagent_type": "weather"}},
            {"name": "get_weather", "args": {}},
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_defined_subagent_custom_model(self):
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "system_prompt": "Use the get_weather tool to get the weather in a city.",
                            "tools": [get_weather],
                            "model": "gpt-5.4",
                        }
                    ],
                )
            ],
        )
        expected_tool_calls = [
            {
                "name": "task",
                "args": {"subagent_type": "weather"},
                "model": "claude-sonnet-4-6",
            },
            {"name": "get_weather", "args": {}, "model": "gpt-5.4"},
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_defined_subagent_custom_middleware(self):
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "system_prompt": "Use the get_weather tool to get the weather in a city.",
                            "tools": [],  # No tools, only in middleware
                            "model": "gpt-5.4",
                            "middleware": [WeatherMiddleware()],
                        }
                    ],
                )
            ],
        )
        expected_tool_calls = [
            {
                "name": "task",
                "args": {"subagent_type": "weather"},
                "model": "claude-sonnet-4-6",
            },
            {"name": "get_weather", "args": {}, "model": "gpt-5.4"},
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_defined_subagent_custom_runnable(self):
        custom_subagent = create_agent(
            model="gpt-5.4",
            system_prompt="Use the get_weather tool to get the weather in a city.",
            tools=[get_weather],
        )
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the task tool to call a subagent.",
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        {
                            "name": "weather",
                            "description": "This subagent can get weather in cities.",
                            "runnable": custom_subagent,
                        }
                    ],
                )
            ],
        )
        expected_tool_calls = [
            {
                "name": "task",
                "args": {"subagent_type": "weather"},
                "model": "claude-sonnet-4-6",
            },
            {"name": "get_weather", "args": {}, "model": "gpt-5.4"},
        ]
        assert_expected_subgraph_actions(
            expected_tool_calls,
            agent,
            {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
        )

    def test_subagent_response_format_serialized_as_json(self):
        """Test that subagent responseFormat produces JSON-serialized ToolMessage content.

        Verifies the end-to-end flow when `response_format` is set directly on a
        `SubAgent` spec: the subagent's `structured_response` is JSON-serialized
        into the ToolMessage content returned to the parent agent.
        """

        class SubagentFindings(BaseModel):
            findings: str = Field(description="The findings")
            confidence: float = Field(description="Confidence score")
            summary: str = Field(description="Brief summary")

        agent = create_agent(
            model="claude-sonnet-4-20250514",
            system_prompt="You are an orchestrator. Always delegate tasks to the appropriate subagent via the task tool.",
            middleware=[
                SubAgentMiddleware(
                    subagents=[
                        {
                            "name": "foo",
                            "description": "Call this when the user says 'foo'",
                            "system_prompt": "You are a foo agent",
                            "model": "claude-haiku-4-5",
                            "tools": [],
                            "response_format": ToolStrategy(schema=SubagentFindings),
                        },
                    ],
                )
            ],
        )

        result = agent.invoke(
            {"messages": [HumanMessage(content="foo - tell me how confident you are that pineapple belongs on pizza")]},
            {"recursion_limit": 100},
        )

        agent_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        tool_calls = [tc for msg in agent_messages for tc in (msg.tool_calls or [])]
        assert any(tc["name"] == "task" and tc["args"].get("subagent_type") == "foo" for tc in tool_calls)

        task_tool_messages = [msg for msg in result["messages"] if msg.type == "tool" and msg.name == "task"]
        assert len(task_tool_messages) > 0

        task_tool_message = task_tool_messages[0]
        parsed = json.loads(task_tool_message.content)
        assert "findings" in parsed
        assert "confidence" in parsed
        assert "summary" in parsed
        assert isinstance(parsed["findings"], str)
        assert isinstance(parsed["confidence"], (int, float))
        assert isinstance(parsed["summary"], str)

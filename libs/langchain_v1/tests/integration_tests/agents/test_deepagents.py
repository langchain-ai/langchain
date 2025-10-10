from langchain.agents.deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.tools import tool, InjectedToolCallId
from typing import Annotated
from langchain.tools.tool_node import InjectedState
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.types import Command
from langchain_core.messages import ToolMessage


def assert_all_deepagent_qualities(agent):
    assert "todos" in agent.stream_channels
    assert "files" in agent.stream_channels
    assert "write_todos" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "ls" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "read_file" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "write_file" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "edit_file" in agent.nodes["tools"].bound._tools_by_name.keys()
    assert "task" in agent.nodes["tools"].bound._tools_by_name.keys()


SAMPLE_MODEL = "claude-3-5-sonnet-20240620"


@tool(description="Use this tool to get the weather")
def get_weather(location: str):
    return f"The weather in {location} is sunny."


@tool(description="Use this tool to get the latest soccer scores")
def get_soccer_scores(team: str):
    return f"The latest soccer scores for {team} are 2-1."


@tool(description="Sample tool")
def sample_tool(sample_input: str):
    return sample_input


@tool(description="Sample tool with injected state")
def sample_tool_with_injected_state(sample_input: str, state: Annotated[dict, InjectedState]):
    return sample_input + state["sample_input"]


TOY_BASKETBALL_RESEARCH = "Lebron James is the best basketball player of all time with over 40k points and 21 seasons in the NBA."


@tool(description="Use this tool to conduct research into basketball and save it to state")
def research_basketball(
    topic: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    current_research = state.get("research", "")
    research = f"{current_research}\n\nResearching on {topic}... Done! {TOY_BASKETBALL_RESEARCH}"
    return Command(
        update={
            "research": research,
            "messages": [ToolMessage(research, tool_call_id=tool_call_id)],
        }
    )


class ResearchState(AgentState):
    research: str


class ResearchMiddlewareWithTools(AgentMiddleware):
    state_schema = ResearchState
    tools = [research_basketball]


class ResearchMiddleware(AgentMiddleware):
    state_schema = ResearchState


class SampleMiddlewareWithTools(AgentMiddleware):
    tools = [sample_tool]


class SampleState(AgentState):
    sample_input: str


class SampleMiddlewareWithToolsAndState(AgentMiddleware):
    state_schema = SampleState
    tools = [sample_tool]


class WeatherToolMiddleware(AgentMiddleware):
    tools = [get_weather]


class TestDeepAgentsFilesystem:
    def test_base_deep_agent(self):
        agent = create_deep_agent()
        assert_all_deepagent_qualities(agent)

    def test_deep_agent_with_tool(self):
        agent = create_deep_agent(tools=[sample_tool])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name.keys()

    def test_deep_agent_with_middleware_with_tool(self):
        agent = create_deep_agent(middleware=[SampleMiddlewareWithTools()])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name.keys()

    def test_deep_agent_with_middleware_with_tool_and_state(self):
        agent = create_deep_agent(middleware=[SampleMiddlewareWithToolsAndState()])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name.keys()
        assert "sample_input" in agent.stream_channels

    def test_deep_agent_with_subagents(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "prompt": "You are a weather agent.",
                "tools": [get_weather],
                "model": SAMPLE_MODEL,
            }
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}
        )
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(
            [
                tool_call["name"] == "task"
                and tool_call["args"].get("subagent_type") == "weather_agent"
                for tool_call in tool_calls
            ]
        )

    def test_deep_agent_with_subagents_gen_purpose(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "prompt": "You are a weather agent.",
                "tools": [get_weather],
                "model": SAMPLE_MODEL,
            }
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the general purpose subagent to call the sample tool",
                    }
                ]
            }
        )
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(
            [
                tool_call["name"] == "task"
                and tool_call["args"].get("subagent_type") == "general-purpose"
                for tool_call in tool_calls
            ]
        )

    def test_deep_agent_with_subagents_with_middleware(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "prompt": "You are a weather agent.",
                "tools": [],
                "model": SAMPLE_MODEL,
                "middleware": [WeatherToolMiddleware()],
            }
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}
        )
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(
            [
                tool_call["name"] == "task"
                and tool_call["args"].get("subagent_type") == "weather_agent"
                for tool_call in tool_calls
            ]
        )

    def test_deep_agent_with_custom_subagents(self):
        subagents = [
            {
                "name": "weather_agent",
                "description": "Use this agent to get the weather",
                "prompt": "You are a weather agent.",
                "tools": [get_weather],
                "model": SAMPLE_MODEL,
            },
            {
                "name": "soccer_agent",
                "description": "Use this agent to get the latest soccer scores",
                "runnable": create_agent(
                    model=SAMPLE_MODEL,
                    tools=[get_soccer_scores],
                    system_prompt="You are a soccer agent.",
                ),
            },
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Look up the weather in Tokyo, and the latest scores for Manchester City!",
                    }
                ]
            }
        )
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(
            [
                tool_call["name"] == "task"
                and tool_call["args"].get("subagent_type") == "weather_agent"
                for tool_call in tool_calls
            ]
        )
        assert any(
            [
                tool_call["name"] == "task"
                and tool_call["args"].get("subagent_type") == "soccer_agent"
                for tool_call in tool_calls
            ]
        )

    def test_deep_agent_with_extended_state_and_subagents(self):
        subagents = [
            {
                "name": "basketball_info_agent",
                "description": "Use this agent to get surface level info on any basketball topic",
                "prompt": "You are a basketball info agent.",
                "middleware": [ResearchMiddlewareWithTools()],
            }
        ]
        agent = create_deep_agent(
            tools=[sample_tool], subagents=subagents, middleware=[ResearchMiddleware()]
        )
        assert_all_deepagent_qualities(agent)
        assert "research" in agent.stream_channels
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Get surface level info on lebron james"}]},
            config={"recursion_limit": 100},
        )
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(
            [
                tool_call["name"] == "task"
                and tool_call["args"].get("subagent_type") == "basketball_info_agent"
                for tool_call in tool_calls
            ]
        )
        assert TOY_BASKETBALL_RESEARCH in result["research"]

    def test_deep_agent_with_subagents_no_tools(self):
        subagents = [
            {
                "name": "basketball_info_agent",
                "description": "Use this agent to get surface level info on any basketball topic",
                "prompt": "You are a basketball info agent.",
            }
        ]
        agent = create_deep_agent(tools=[sample_tool], subagents=subagents)
        assert_all_deepagent_qualities(agent)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the basketball info subagent to call the sample tool",
                    }
                ]
            },
            config={"recursion_limit": 100},
        )
        agent_messages = [msg for msg in result.get("messages", []) if msg.type == "ai"]
        tool_calls = [tool_call for msg in agent_messages for tool_call in msg.tool_calls]
        assert any(
            [
                tool_call["name"] == "task"
                and tool_call["args"].get("subagent_type") == "basketball_info_agent"
                for tool_call in tool_calls
            ]
        )

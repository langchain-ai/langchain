"""Test that RunnableConfig is passed to tools in AgentExecutor."""

from typing import Optional
from langchain_core.tools import BaseTool
from langchain_core.runnables.config import RunnableConfig
from langchain.agents import AgentExecutor
from langchain_core.language_models.fake import FakeListLLM
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate


class ConfigCaptureTool(BaseTool):
    """Tool that captures RunnableConfig when called."""

    name: str = "config_capture_tool"
    description: str = "A tool that captures the config parameter"
    captured_config: Optional[RunnableConfig] = None

    def _run(
        self,
        query: str,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> str:
        """Capture the config parameter."""
        ConfigCaptureTool.captured_config = config
        return f"Captured config: {config is not None}"


def test_runnable_config_passed_to_tools() -> None:
    """Test that RunnableConfig is properly passed to tools."""

    # Reset captured config
    ConfigCaptureTool.captured_config = None

    # Create tool
    tool = ConfigCaptureTool()

    # Create agent with fake LLM that will call our tool
    llm = FakeListLLM(
        responses=[
            "I should use the config_capture_tool.\nAction: config_capture_tool\nAction Input: test",
            "Final Answer: Tool has been executed",
        ]
    )

    # Create the prompt template
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # Create agent and executor
    agent = create_react_agent(llm, [tool], prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=False)

    # Test with RunnableConfig
    test_config: RunnableConfig = {
        "configurable": {"test_key": "test_value"},
        "metadata": {"source": "test"},
    }

    # Execute agent
    agent_executor.invoke({"input": "Use the config capture tool"}, config=test_config)

    # Assertion
    assert (
        ConfigCaptureTool.captured_config is not None
    ), "RunnableConfig should be passed to tools"

"""Integration test for Dall-E image generator agent."""
from langchain_xfyun.agents import AgentType, initialize_agent, load_tools
from langchain_xfyun.llms import OpenAI


def test_call() -> None:
    """Test that the agent runs and returns output."""
    llm = OpenAI(temperature=0.9)
    tools = load_tools(["dalle-image-generator"])

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    output = agent.run("Create an image of a volcano island")
    assert output is not None

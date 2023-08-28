from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_xfyun.agents.agent_types import AgentType
from langchain_xfyun.agents.initialize import initialize_agent, load_agent
from langchain_xfyun.agents.tools import Tool
from langchain_xfyun.llms.fake import FakeListLLM


def test_mrkl_serialization() -> None:
    agent = initialize_agent(
        [
            Tool(
                name="Test tool",
                func=lambda x: x,
                description="Test description",
            )
        ],
        FakeListLLM(responses=[]),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    with TemporaryDirectory() as tempdir:
        file = Path(tempdir) / "agent.json"
        agent.save_agent(file)
        load_agent(file)

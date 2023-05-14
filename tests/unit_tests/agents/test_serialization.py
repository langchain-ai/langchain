from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent, load_agent
from langchain.llms.fake import FakeListLLM


def test_mrkl_serialization() -> None:
    agent = initialize_agent(
        [],
        FakeListLLM(responses=[]),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    with TemporaryDirectory() as tempdir:
        file = Path(tempdir) / "agent.json"
        agent.save_agent(file)
        load_agent(file)

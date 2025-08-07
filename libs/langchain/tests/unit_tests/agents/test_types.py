from langchain.agents.agent_types import AgentType
from langchain.agents.types import AGENT_TO_CLASS


def test_confirm_full_coverage() -> None:
    assert list(AgentType) == list(AGENT_TO_CLASS.keys())

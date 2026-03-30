from langchain_a2a.client import MultiAgentA2AClient
from langchain_a2a.tools import get_tools


def test_get_tools_empty() -> None:
    client = MultiAgentA2AClient(endpoint="http://test")
    tools = get_tools(client)
    assert tools == []

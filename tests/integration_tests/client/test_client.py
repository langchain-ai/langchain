"""LangChain+ langchain_client Integration Tests."""
from uuid import uuid4

import pytest
from tenacity import RetryError

from langchain.callbacks.manager import tracing_v2_enabled
from langchain.client import LangChainPlusClient
from langchain.tools.base import tool


@pytest.fixture
def langchain_client(monkeypatch: pytest.MonkeyPatch) -> LangChainPlusClient:
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "http://localhost:1984")
    return LangChainPlusClient()


def test_sessions(
    langchain_client: LangChainPlusClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test sessions."""
    session_names = set([session.name for session in langchain_client.list_sessions()])
    new_session = f"Session {uuid4()}"
    assert new_session not in session_names

    @tool
    def example_tool() -> str:
        """Call me, maybe."""
        return "test_tool"

    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "http://localhost:1984")
    with tracing_v2_enabled(session_name=new_session):
        example_tool({})
    session = langchain_client.read_session(session_name=new_session)
    assert session.name == new_session
    session_names = set([sess.name for sess in langchain_client.list_sessions()])
    assert new_session in session_names
    runs = list(langchain_client.list_runs(session_name=new_session))
    session_id_runs = list(langchain_client.list_runs(session_id=session.id))
    assert len(runs) == len(session_id_runs) == 1
    assert runs[0].id == session_id_runs[0].id
    langchain_client.delete_session(session_name=new_session)

    with pytest.raises(RetryError):
        langchain_client.read_session(session_name=new_session)
    assert new_session not in set(
        [sess.name for sess in langchain_client.list_sessions()]
    )
    with pytest.raises(RetryError):
        langchain_client.delete_session(session_name=new_session)
    with pytest.raises(RetryError):
        langchain_client.read_run(run_id=str(runs[0].id))

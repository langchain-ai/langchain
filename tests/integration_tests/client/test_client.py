"""LangChain+ langchain_client Integration Tests."""
import os
from uuid import uuid4

import pytest

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks.manager import tracing_v2_enabled
from langchain.chat_models import ChatOpenAI
from langchain.client import LangChainPlusClient


@pytest.fixture
def langchain_client(monkeypatch: pytest.MonkeyPatch) -> LangChainPlusClient:
    monkeypatch.setenv("LANGCHAIN_ENDPOINT", "http://localhost:8000")
    return LangChainPlusClient()


def test_feedback_cycle(
    monkeypatch: pytest.MonkeyPatch, langchain_client: LangChainPlusClient
) -> None:
    """Test that feedback is correctly created and updated."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_SESSION", f"Feedback Testing {uuid4()}")
    llm = ChatOpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
    )

    agent.run(
        "What is the population of Kuala Lumpur as of January, 2023?"
        " What is it's square root?"
    )
    other_session_name = f"Feedback Testing {uuid4()}"
    with tracing_v2_enabled(session_name=other_session_name):
        try:
            agent.run("What is the square root of 3?")
        except Exception as e:
            print(e)
    runs = list(
        langchain_client.list_runs(
            session_name=os.environ["LANGCHAIN_SESSION"], error=False, execution_order=1
        )
    )
    assert len(runs) == 1
    order_2 = list(
        langchain_client.list_runs(
            session_name=os.environ["LANGCHAIN_SESSION"], execution_order=2
        )
    )
    assert len(order_2) > 0
    langchain_client.create_feedback(str(order_2[0].id), "test score", metric_value=0)
    feedback = langchain_client.create_feedback(
        str(runs[0].id), "test score", metric_value=1
    )
    feedbacks = list(langchain_client.list_feedback(run_ids=[str(runs[0].id)]))
    assert len(feedbacks) == 1
    assert feedbacks[0].id == feedback.id

    # Add feedback to other session
    other_runs = list(
        langchain_client.list_runs(session_name=other_session_name, execution_order=1)
    )
    assert len(other_runs) == 1
    langchain_client.create_feedback(
        run_id=str(other_runs[0].id), metric_name="test score", metric_value=0
    )
    all_runs = list(
        langchain_client.list_runs(session_name=os.environ["LANGCHAIN_SESSION"])
    ) + list(langchain_client.list_runs(session_name=other_session_name))
    test_run_ids = [str(run.id) for run in all_runs]
    all_feedback = list(langchain_client.list_feedback(run_ids=test_run_ids))
    assert len(all_feedback) == 3
    for feedback in all_feedback:
        langchain_client.delete_feedback(str(feedback.id))
    feedbacks = list(langchain_client.list_feedback(run_ids=test_run_ids))
    assert len(feedbacks) == 0

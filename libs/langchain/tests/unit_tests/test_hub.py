from typing import Any, List
from unittest.mock import MagicMock, Mock, patch

from langchain_core.load.dump import dumps
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableBinding
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

from langchain import hub


class FakeTracer(BaseTracer):
    """Fake tracer that records LangChain execution."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: List[Run] = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        self.runs.append(run)


repo_dict = {
    "wfh/my-prompt-1": ChatPromptTemplate.from_messages(
        [("system", "a"), ("user", "1")]
    ),
    "wfh/my-random-object": {"Hi": "there"},
}


def repo_lookup(owner_repo_commit: str, **kwargs: Any) -> ChatPromptTemplate:
    return dumps(repo_dict[owner_repo_commit])


@patch("langchain.hub._get_client")
def test_hub_pull_metadata(mock_get_client: Mock) -> None:
    mock_client = MagicMock()
    mock_client.pull = repo_lookup
    mock_get_client.return_value = mock_client
    res = hub.pull("wfh/my-prompt-1")
    assert isinstance(res, RunnableBinding)
    tracer = FakeTracer()
    result = res.invoke({}, {"callbacks": [tracer]})
    assert result.messages[0].content == "a"
    assert result.messages[1].content == "1"
    assert len(tracer.runs) == 1
    run = tracer.runs[0]
    assert run.extra["metadata"]["hub_owner_repo_commit"] == "wfh/my-prompt-1"


@patch("langchain.hub._get_client")
def test_hub_pull_random_object(mock_get_client: Mock) -> None:
    mock_client = MagicMock()
    mock_client.pull = repo_lookup
    mock_get_client.return_value = mock_client
    res = hub.pull("wfh/my-random-object")
    assert res == {"Hi": "there"}

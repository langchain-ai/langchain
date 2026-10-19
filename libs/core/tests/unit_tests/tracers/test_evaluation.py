"""Unit tests for `EvaluatorCallbackHandler` feedback forwarding."""

import uuid
from unittest.mock import MagicMock

import langsmith
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import FeedbackConfig, Run

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler


def _make_handler() -> tuple[EvaluatorCallbackHandler, MagicMock]:
    """Create a handler backed by a mock LangSmith client."""
    client = MagicMock(spec=langsmith.Client)
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)
    return handler, client


def _make_run() -> MagicMock:
    """Create a mock run with a UUID id."""
    run = MagicMock(spec=Run)
    run.id = uuid.uuid4()
    return run


def test_log_evaluation_feedback_forwards_feedback_config() -> None:
    """A dict `feedback_config` set on the result must reach `create_feedback`."""
    handler, client = _make_handler()
    result = EvaluationResult(
        key="sentiment",
        score=0.75,
        comment="mostly positive",
        feedback_config={"type": "continuous", "min": 0, "max": 1},
    )

    handler._log_evaluation_feedback(result, _make_run())

    kwargs = client.create_feedback.call_args.kwargs
    assert kwargs["feedback_config"] == {"type": "continuous", "min": 0, "max": 1}
    assert kwargs["score"] == 0.75
    assert kwargs["comment"] == "mostly positive"


def test_log_evaluation_feedback_forwards_feedback_config_instance() -> None:
    """A `FeedbackConfig` instance is forwarded unchanged."""
    handler, client = _make_handler()
    config = FeedbackConfig(type="continuous", min=0, max=1)
    result = EvaluationResult(key="sentiment", score=0.5, feedback_config=config)

    handler._log_evaluation_feedback(result, _make_run())

    assert client.create_feedback.call_args.kwargs["feedback_config"] == config

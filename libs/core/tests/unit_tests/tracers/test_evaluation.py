"""Tests for EvaluatorCallbackHandler feedback_config forwarding."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.create_feedback = MagicMock()
    return client


def _make_handler(mock_client: MagicMock) -> EvaluatorCallbackHandler:
    with patch(
        "langchain_core.tracers.evaluation.langchain_tracer.get_client",
        return_value=mock_client,
    ):
        return EvaluatorCallbackHandler(
            evaluators=[],
            client=mock_client,
            max_concurrency=0,
        )


def test_log_evaluation_feedback_forwards_feedback_config(
    mock_client: MagicMock,
) -> None:
    """feedback_config from EvaluationResult should be forwarded to create_feedback."""
    handler = _make_handler(mock_client)
    feedback_config = {"type": "continuous", "min": 0, "max": 1}
    result = EvaluationResult(
        key="accuracy",
        score=0.95,
        feedback_config=feedback_config,
    )

    run = MagicMock()
    run.id = uuid4()

    handler._log_evaluation_feedback(result, run)

    mock_client.create_feedback.assert_called_once()
    call_kwargs = mock_client.create_feedback.call_args
    assert call_kwargs.kwargs.get("feedback_config") == feedback_config


def test_log_evaluation_feedback_none_feedback_config(
    mock_client: MagicMock,
) -> None:
    """When feedback_config is None it should still be passed through."""
    handler = _make_handler(mock_client)
    result = EvaluationResult(key="relevance", score=0.8)

    run = MagicMock()
    run.id = uuid4()

    handler._log_evaluation_feedback(result, run)

    mock_client.create_feedback.assert_called_once()
    call_kwargs = mock_client.create_feedback.call_args
    assert call_kwargs.kwargs.get("feedback_config") is None

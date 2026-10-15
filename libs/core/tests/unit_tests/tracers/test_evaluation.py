"""Unit tests for `EvaluatorCallbackHandler` feedback logging.

Regression tests for langchain-ai/langchain#31802: an evaluator could set
`feedback_config` on an `EvaluationResult`, but
`EvaluatorCallbackHandler._log_evaluation_feedback` never forwarded it to
`Client.create_feedback`, so the config was silently dropped before it
reached LangSmith.
"""

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from langsmith.evaluation.evaluator import FeedbackConfig

from langchain_core.tracers.evaluation import (
    EvaluationResult,
    EvaluatorCallbackHandler,
)


def _make_handler() -> tuple[EvaluatorCallbackHandler, MagicMock]:
    mock_client = MagicMock()
    handler = EvaluatorCallbackHandler(
        evaluators=[],
        client=mock_client,
        max_concurrency=0,
    )
    return handler, mock_client


def _log_and_get_kwargs(
    handler: EvaluatorCallbackHandler, mock_client: MagicMock, result: Any
) -> dict[str, Any]:
    fake_run = MagicMock()
    fake_run.id = uuid4()
    handler._log_evaluation_feedback(result, fake_run)
    assert mock_client.create_feedback.call_count == 1
    return mock_client.create_feedback.call_args.kwargs


def test_feedback_config_dict_with_valid_fields_is_forwarded() -> None:
    """A valid `feedback_config` dict reaches `create_feedback` unchanged."""
    handler, mock_client = _make_handler()
    config = {"type": "continuous", "min": 0.0, "max": 1.0}
    result = EvaluationResult(
        key="sentiment", score=1.0, value="positive", feedback_config=config
    )
    kwargs = _log_and_get_kwargs(handler, mock_client, result)
    assert kwargs["feedback_config"] == config


def test_feedback_config_categorical_is_forwarded() -> None:
    handler, mock_client = _make_handler()
    config = {
        "type": "categorical",
        "categories": [{"label": "positive"}, {"label": "negative"}],
    }
    result = EvaluationResult(key="sentiment", value="positive", feedback_config=config)
    kwargs = _log_and_get_kwargs(handler, mock_client, result)
    assert kwargs["feedback_config"] == config


def test_feedback_config_with_unknown_keys_is_forwarded_not_dropped() -> None:
    """Unknown keys must not be silently dropped at the callback boundary.

    LangSmith owns validation/normalization of `feedback_config`; the
    handler's job is to forward the value unchanged rather than discard it
    (the original #31802 report).
    """
    handler, mock_client = _make_handler()
    result = EvaluationResult(
        key="sentiment", value="positive", feedback_config={"threshold": 1.0}
    )
    # The config survives construction on current langsmith versions.
    assert result.feedback_config == {"threshold": 1.0}
    kwargs = _log_and_get_kwargs(handler, mock_client, result)
    assert kwargs["feedback_config"] == {"threshold": 1.0}


def test_feedback_config_instance_is_forwarded() -> None:
    """A `FeedbackConfig` instance (not just a dict) is forwarded."""
    handler, mock_client = _make_handler()
    config = FeedbackConfig(type="continuous", min=0.0, max=1.0)
    result = EvaluationResult(key="score", score=0.5, feedback_config=config)
    kwargs = _log_and_get_kwargs(handler, mock_client, result)
    assert kwargs["feedback_config"] == config


def test_feedback_config_none_is_forwarded_as_none() -> None:
    """When no config is set, `create_feedback` receives `feedback_config=None`."""
    handler, mock_client = _make_handler()
    result = EvaluationResult(key="sentiment", score=1.0, value="positive")
    assert result.feedback_config is None
    kwargs = _log_and_get_kwargs(handler, mock_client, result)
    assert kwargs["feedback_config"] is None


def test_other_fields_still_forwarded_alongside_feedback_config() -> None:
    handler, mock_client = _make_handler()
    result = EvaluationResult(
        key="sentiment",
        score=1.0,
        value="positive",
        comment="looks good",
        feedback_config={"type": "continuous", "min": 0.0, "max": 1.0},
    )
    kwargs = _log_and_get_kwargs(handler, mock_client, result)
    assert kwargs["score"] == 1.0
    assert kwargs["value"] == "positive"
    assert kwargs["comment"] == "looks good"
    assert kwargs["feedback_config"] == {"type": "continuous", "min": 0.0, "max": 1.0}
    # Positional args: run_id, key
    args = mock_client.create_feedback.call_args.args
    assert args[1] == "sentiment"

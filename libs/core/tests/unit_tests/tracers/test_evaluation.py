"""Regression tests for EvaluationResult.feedback_config forwarding.

Ensures that EvaluatorCallbackHandler._log_evaluation_feedback forwards
feedback_config to client.create_feedback (see issue #31802).
"""

import unittest.mock
from uuid import UUID

import langsmith
from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler
from langchain_core.tracers.schemas import Run


def _make_run() -> Run:
    """Create a minimal Run for testing."""
    return Run(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        name="test_run",
        run_type="chain",
        inputs={},
        trace_id=UUID("00000000-0000-0000-0000-000000000002"),
        dotted_order="1.2",
    )


def test_feedback_config_forwarded() -> None:
    """Test that feedback_config is forwarded to create_feedback."""
    client = unittest.mock.MagicMock(spec=langsmith.Client)
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)

    run = _make_run()
    result = EvaluationResult(
        key="quality",
        score=0.9,
        feedback_config={"type": "continuous", "min": 0.0, "max": 1.0},
    )

    handler._log_evaluation_feedback(result, run)

    client.create_feedback.assert_called_once()
    call_kwargs = client.create_feedback.call_args
    assert call_kwargs[1]["feedback_config"] == {
        "type": "continuous",
        "min": 0.0,
        "max": 1.0,
    }


def test_feedback_config_none_forwarded() -> None:
    """Test that feedback_config=None is forwarded when not set."""
    client = unittest.mock.MagicMock(spec=langsmith.Client)
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)

    run = _make_run()
    result = EvaluationResult(key="quality", score=0.9)

    handler._log_evaluation_feedback(result, run)

    client.create_feedback.assert_called_once()
    call_kwargs = client.create_feedback.call_args
    assert call_kwargs[1]["feedback_config"] is None


def test_other_fields_still_forwarded() -> None:
    """Test that all other fields are still forwarded correctly."""
    client = unittest.mock.MagicMock(spec=langsmith.Client)
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)

    run = _make_run()
    result = EvaluationResult(
        key="quality",
        score=0.9,
        value="good",
        comment="Looks great",
        correction={"suggestion": "improve"},
        evaluator_info={"model": "gpt-4"},
        source_run_id=UUID("00000000-0000-0000-0000-000000000003"),
        feedback_config={"type": "categorical", "categories": ["good", "bad"]},
    )

    handler._log_evaluation_feedback(result, run)

    client.create_feedback.assert_called_once()
    call_kwargs = client.create_feedback.call_args
    assert call_kwargs[1]["score"] == 0.9
    assert call_kwargs[1]["value"] == "good"
    assert call_kwargs[1]["comment"] == "Looks great"
    assert call_kwargs[1]["correction"] == {"suggestion": "improve"}
    assert call_kwargs[1]["source_info"] == {"model": "gpt-4"}
    assert call_kwargs[1]["feedback_config"] == {
        "type": "categorical",
        "categories": ["good", "bad"],
    }


def test_multiple_results_forward_feedback_config() -> None:
    """Test that feedback_config is forwarded for multiple results."""
    client = unittest.mock.MagicMock(spec=langsmith.Client)
    handler = EvaluatorCallbackHandler(evaluators=[], client=client)

    run = _make_run()
    results = {
        "results": [
            EvaluationResult(
                key="quality",
                score=0.9,
                feedback_config={"type": "continuous", "min": 0.0, "max": 1.0},
            ),
            EvaluationResult(
                key="relevance",
                score=0.8,
                feedback_config={"type": "continuous", "min": 0.0, "max": 1.0},
            ),
        ]
    }

    handler._log_evaluation_feedback(results, run)

    assert client.create_feedback.call_count == 2
    first_call = client.create_feedback.call_args_list[0][1]
    second_call = client.create_feedback.call_args_list[1][1]
    assert first_call["feedback_config"] == {
        "type": "continuous",
        "min": 0.0,
        "max": 1.0,
    }
    assert second_call["feedback_config"] == {
        "type": "continuous",
        "min": 0.0,
        "max": 1.0,
    }

from unittest.mock import MagicMock
from uuid import UUID

import langsmith.schemas
from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler


def _mock_handler() -> tuple[EvaluatorCallbackHandler, MagicMock]:
    client = MagicMock()
    handler = EvaluatorCallbackHandler(
        evaluators=(),
        client=client,
        max_concurrency=0,
    )
    return handler, client


def test_feedback_config_forwarded_to_create_feedback() -> None:
    handler, client = _mock_handler()
    run_id = UUID(int=1)
    evaluation_result = EvaluationResult(
        key="sentiment",
        score=0.8,
        value="positive",
        comment="Great",
        correction={"name": "happy"},
        source_run_id=UUID(int=2),
        feedback_config={"type": "continuous", "min": 0, "max": 1},
    )
    run = MagicMock()
    run.id = run_id

    handler._log_evaluation_feedback(evaluation_result, run, source_run_id=run_id)

    client.create_feedback.assert_called_once_with(
        run_id,
        "sentiment",
        score=0.8,
        value="positive",
        comment="Great",
        correction={"name": "happy"},
        source_info={},
        source_run_id=UUID(int=2),
        feedback_source_type=langsmith.schemas.FeedbackSourceType.MODEL,
        feedback_config={"type": "continuous", "min": 0, "max": 1},
    )


def test_feedback_config_none_when_unset() -> None:
    handler, client = _mock_handler()
    evaluation_result = EvaluationResult(key="sentiment", score=0.8)
    run = MagicMock()

    handler._log_evaluation_feedback(evaluation_result, run)

    assert client.create_feedback.call_args.kwargs["feedback_config"] is None

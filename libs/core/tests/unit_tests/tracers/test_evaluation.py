from unittest import mock

from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import EvaluatorCallbackHandler


def test_feedback_config_forwarded_to_create_feedback() -> None:
    client = mock.MagicMock()
    handler = EvaluatorCallbackHandler(
        evaluators=[],
        client=client,
    )
    run = mock.MagicMock()
    run.id = mock.sentinel.run_id

    eval_result = EvaluationResult(
        key="sentiment",
        score=1.0,
        feedback_config={"type": "continuous", "min": 0, "max": 1},
    )

    handler._log_evaluation_feedback(eval_result, run)

    client.create_feedback.assert_called_once()
    _, kwargs = client.create_feedback.call_args
    assert kwargs["feedback_config"] == {"type": "continuous", "min": 0, "max": 1}


def test_feedback_config_none_is_not_sent_as_arg() -> None:
    client = mock.MagicMock()
    handler = EvaluatorCallbackHandler(
        evaluators=[],
        client=client,
    )
    run = mock.MagicMock()
    run.id = mock.sentinel.run_id

    eval_result = EvaluationResult(
        key="sentiment",
        score=1.0,
        feedback_config=None,
    )

    handler._log_evaluation_feedback(eval_result, run)

    client.create_feedback.assert_called_once()
    _, kwargs = client.create_feedback.call_args
    assert kwargs["feedback_config"] is None


def test_feedback_config_with_arbitrary_dict() -> None:
    client = mock.MagicMock()
    handler = EvaluatorCallbackHandler(
        evaluators=[],
        client=client,
    )
    run = mock.MagicMock()
    run.id = mock.sentinel.run_id

    eval_result = EvaluationResult(
        key="sentiment",
        score=1.0,
        feedback_config={"threshold": 0.5, "custom_key": "value"},
    )

    handler._log_evaluation_feedback(eval_result, run)

    client.create_feedback.assert_called_once()
    _, kwargs = client.create_feedback.call_args
    assert kwargs["feedback_config"] == {"threshold": 0.5, "custom_key": "value"}

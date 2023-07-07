"""Test the loading function for evalutors."""

from unittest.mock import MagicMock

import pytest

from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.evaluation.loading import load_evaluators
from langchain.evaluation.run_evaluators.string_run_evaluator import (
    StringRunEvaluatorChain,
)
from langchain.evaluation.schema import StringEvaluator
from tests.unit_tests.chains.test_base import FakeChain
from tests.unit_tests.llms.fake_chat_model import FakeChatModel
from tests.unit_tests.llms.fake_llm import FakeLLM


@pytest.mark.parametrize("evaluator_type", ["qa", "cot_qa", "context_qa", "criteria"])
def test_load_string_run_evaluators_with_llm(evaluator_type: str) -> None:
    """Test loading evaluators."""
    fake_llm = FakeLLM(
        queries={"text": "The meaning of life\nCORRECT"}, sequential_responses=True
    )
    evaluator = load_evaluators([evaluator_type], llm=fake_llm)[0]  # type: ignore
    if not isinstance(evaluator, StringEvaluator):
        raise ValueError("Evaluator is not a string evaluator")
    model = FakeLLM(queries={"text": "Foo output"}, sequential_responses=True)
    kwargs = {}
    if evaluator.requires_reference:
        kwargs["reference_key"] = "generations"
    run_evaluator = StringRunEvaluatorChain.from_model_and_evaluator(
        model, evaluator, **kwargs
    )
    callback = RunCollectorCallbackHandler()
    model.predict("Foo input", callbacks=[callback])
    run = callback.traced_runs[0]
    example = MagicMock()
    example.inputs = {}
    example.outputs = {"generations": "Foo output"}
    result = run_evaluator._prepare_input({"run": run, "example": example})
    assert result["input"] == "Foo input"
    assert result["prediction"] == "Foo output"
    if evaluator.requires_reference:
        assert "reference" in result
        assert result["reference"] == "Foo output"


@pytest.mark.parametrize("evaluator_type", ["qa", "cot_qa", "context_qa", "criteria"])
def test_load_string_run_evaluators_with_chat_model(evaluator_type: str) -> None:
    """Test loading evaluators."""
    fake_llm = FakeLLM(
        queries={"text": "The meaning of life\nCORRECT"}, sequential_responses=True
    )
    evaluator = load_evaluators([evaluator_type], llm=fake_llm)[0]  # type: ignore
    if not isinstance(evaluator, StringEvaluator):
        raise ValueError("Evaluator is not a string evaluator")
    model = FakeChatModel()
    kwargs = {}
    if evaluator.requires_reference:
        kwargs["reference_key"] = "generations"
    run_evaluator = StringRunEvaluatorChain.from_model_and_evaluator(
        model, evaluator, **kwargs
    )
    callback = RunCollectorCallbackHandler()
    model.predict("Foo input", callbacks=[callback])
    run = callback.traced_runs[0]
    example = MagicMock()
    example.inputs = {}
    example.outputs = {"generations": "Another fake response"}
    result = run_evaluator._prepare_input({"run": run, "example": example})
    assert result["input"] == "Human: Foo input"
    assert result["prediction"] == "AI: fake response"
    if evaluator.requires_reference:
        assert "reference" in result
        assert result["reference"] == "Another fake response"


@pytest.mark.parametrize("evaluator_type", ["qa", "cot_qa", "context_qa", "criteria"])
def test_load_string_run_evaluators_with_chain(evaluator_type: str) -> None:
    model = FakeChain(
        the_input_keys=["an_input", "another_input"],
    )
    fake_llm = FakeChatModel()
    evaluator = load_evaluators([evaluator_type], llm=fake_llm)[0]  # type: ignore
    if not isinstance(evaluator, StringEvaluator):
        raise ValueError("Evaluator is not a string evaluator")
    # No input key
    with pytest.raises(ValueError, match="multiple input keys"):
        StringRunEvaluatorChain.from_model_and_evaluator(model, evaluator)
    with pytest.raises(ValueError, match="does not have specified"):
        StringRunEvaluatorChain.from_model_and_evaluator(
            model, evaluator, input_key="some_input"
        )
    kwargs = {}
    if evaluator.requires_reference:
        kwargs["reference_key"] = "label_column"
    run_evaluator = StringRunEvaluatorChain.from_model_and_evaluator(
        model, evaluator, input_key="an_input", **kwargs
    )
    callback = RunCollectorCallbackHandler()
    model(
        {"an_input": "Foo input", "another_input": "Another fake response"},
        callbacks=[callback],
    )
    run = callback.traced_runs[0]
    example = MagicMock()
    example.inputs = {}
    example.outputs = {"label_column": "Another fake response"}
    result = run_evaluator._prepare_input({"run": run, "example": example})
    assert result["input"] == "Foo input"
    assert result["prediction"] == "baz"
    if evaluator.requires_reference:
        assert "reference" in result
        assert result["reference"] == "Another fake response"

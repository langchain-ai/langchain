"""Test the loading function for evalutors."""

from unittest.mock import MagicMock

import pytest

from langchain.agents.initialize import initialize_agent
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.evaluation.loading import load_evaluator, load_evaluators
from langchain.evaluation.run_evaluators.agent_trajectory_run_evaluator import (
    AgentTrajectoryRunEvaluatorChain,
)
from langchain.evaluation.run_evaluators.loading import load_run_evaluator_for_model
from langchain.evaluation.run_evaluators.string_run_evaluator import (
    StringRunEvaluatorChain,
)
from langchain.evaluation.schema import AgentTrajectoryEvaluator, StringEvaluator
from langchain.tools.base import tool
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
    evaluator = load_evaluator(evaluator_type, llm=fake_llm)  # type: ignore
    if not isinstance(evaluator, StringEvaluator):
        raise ValueError("Evaluator is not a string evaluator")
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


@pytest.mark.parametrize("requires_reference", [False, True])
@pytest.mark.parametrize("evaluator_type", ["trajectory"])
def test_load_agent_trajectory_evaluator_with_chain(
    evaluator_type: str, requires_reference: bool
) -> None:
    model = FakeChain(
        the_input_keys=["some_input", "another_input"],
    )
    fake_llm = FakeChatModel()
    evaluator = load_evaluators(
        [evaluator_type],  # type: ignore
        llm=fake_llm,
        requires_reference=requires_reference,
    )[0]
    if not isinstance(evaluator, AgentTrajectoryEvaluator):
        raise ValueError("Evaluator is not an agent trajectory evaluator")
    with pytest.raises(ValueError, match="does not have specified prediction_key"):
        AgentTrajectoryRunEvaluatorChain.from_model_and_evaluator(model, evaluator)
    with pytest.raises(ValueError, match="does not have specified input key: 'input'"):
        AgentTrajectoryRunEvaluatorChain.from_model_and_evaluator(
            model, evaluator, prediction_key="bar"
        )
    kwargs = {}
    if evaluator.requires_reference:
        kwargs["reference_key"] = "label_column"
    run_evaluator = AgentTrajectoryRunEvaluatorChain.from_model_and_evaluator(
        model, evaluator, input_key="some_input", prediction_key="bar", **kwargs
    )
    callback = RunCollectorCallbackHandler()
    model(
        {"some_input": "Foo input", "another_input": "Another fake response"},
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


@pytest.mark.parametrize("requires_reference", [False, True])
def test_load_agent_trajectory_evaluator_with_agent_executor(
    requires_reference: bool,
) -> None:
    fake_eval_llm = FakeChatModel()

    @tool
    def fake_tool(txt: str) -> str:
        """Wants to be real."""
        return txt

    fake_llm = FakeLLM(queries={"foo": "Final Answer: pi"}, sequential_responses=True)
    agent_executor = initialize_agent(
        tools=[fake_tool], llm=fake_llm  # type: ignore[list-item]
    )
    run_evaluator = load_run_evaluator_for_model(
        "trajectory",  # type: ignore[arg-type]
        agent_executor,
        eval_llm=fake_eval_llm,
        requires_reference=requires_reference,
    )
    assert isinstance(run_evaluator, AgentTrajectoryRunEvaluatorChain)
    callback = RunCollectorCallbackHandler()
    agent_executor(
        {"input": "This is the input"},
        callbacks=[callback],
    )
    run = callback.traced_runs[0]
    example = MagicMock()
    example.inputs = {}
    example.outputs = {"label_column": "Another fake response"}
    result = run_evaluator._prepare_input({"run": run, "example": example})
    assert result["input"] == "This is the input"
    assert result["prediction"] == "pi"
    if run_evaluator.agent_trajectory_evaluator.requires_reference:
        assert "reference" in result
        assert result["reference"] == "Another fake response"

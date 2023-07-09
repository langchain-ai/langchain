"""Run evaluator wrapper for string evaluators."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchainplus_sdk import EvaluationResult, RunEvaluator
from langchainplus_sdk.schemas import Example, Run
from pydantic import Field

from langchain.agents.agent import AgentExecutor
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.evaluation.run_evaluators.utilities.trajectory_utils import (
    assemble_agent_trajectory,
)
from langchain.evaluation.schema import AgentTrajectoryEvaluator
from langchain.load.serializable import Serializable
from langchain.schema import RUN_KEY
from langchain.schema.agent import AgentAction

logger = logging.getLogger(__name__)


class AgentInput(TypedDict):
    """Values to pass to the trajectory evaluator."""

    input: str
    prediction: str
    agent_trajectory: List[Tuple[AgentAction, str]]
    reference: Optional[str]


class AgentTrajectoryRunMapper(Serializable):
    """Extract Agent Trajectory (and inputs and prediction) to be evaluated."""

    input_key: str = "input"
    """The key from the model Run's inputs to use as the eval input."""
    prediction_key: str = "output"
    """The key from the model Run's outputs to use as the eval prediction."""

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ["prediction", "input", "agent_trajectory"]

    @classmethod
    def from_chain(
        cls,
        model: Chain,
        input_key: Optional[str] = None,
        prediction_key: Optional[str] = None,
    ) -> AgentTrajectoryRunMapper:
        """Create a RunMapper from a chain."""
        error_messages = []
        if input_key is None:
            if len(model.input_keys) > 1:
                error_messages.append(
                    f"Chain '{model.__class__.__name__}' has multiple input"
                    " keys. Please specify input_key as one of"
                    f" {model.input_keys} when loading."
                )
            else:
                input_key = model.input_keys[0]
        elif input_key not in model.input_keys:
            error_messages.append(
                f"Chain '{model.__class__.__name__}' does not have specified"
                f" input key: '{input_key}'."
                f" Must be one of: {model.input_keys}"
            )
        if prediction_key is None:
            if len(model.output_keys) > 1:
                error_messages.append(
                    f"Chain '{model.__class__.__name__}' has multiple"
                    " output keys. Please specify prediction_key"
                    f" as one of {model.output_keys} when loading."
                )
            else:
                prediction_key = model.output_keys[0]
        elif prediction_key not in model.output_keys:
            error_messages.append(
                f"Chain '{model.__class__.__name__}' does not have specified"
                f" prediction_key: '{prediction_key}'."
                f" Must be one of: {model.output_keys}"
            )
        if error_messages:
            raise ValueError("\n".join(error_messages))
        if input_key is None or prediction_key is None:
            # This should never happen, but mypy doesn't know that.
            raise ValueError(
                f"Chain {model.__class__.__name__} has no input or output keys."
            )
        return cls(input_key=input_key, prediction_key=prediction_key)

    def map(self, run: Run) -> AgentInput:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            raise ValueError(f"Run {run.id} has no outputs to evaluate.")
        if run.run_type != "chain":
            raise ValueError(
                f"{self.__class__.__name__} only supports agent (chain) runs."
            )
        if self.input_key not in run.inputs:
            raise ValueError(f"Run {run.id} does not have input key {self.input_key}.")
        elif self.prediction_key not in run.outputs:
            raise ValueError(
                f"Run {run.id} does not have prediction key {self.prediction_key}."
            )
        else:
            agent_trajectory = assemble_agent_trajectory(run)
            return AgentInput(
                input=run.inputs[self.input_key],
                prediction=run.outputs[self.prediction_key],
                agent_trajectory=agent_trajectory,
                reference=None,
            )

    def __call__(self, run: Run) -> AgentInput:
        return self.map(run)


class AgentTrajectoryExampleMapper(Serializable):
    """Map an example, or row in the dataset, to the inputs of an evaluation."""

    reference_key: Optional[str] = None
    """The key in the dataset example row to use as the reference answer."""

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ["reference"]

    def map(self, example: Example) -> Dict[str, str]:
        """Maps the Example, or dataset row to a dictionary."""
        if not example.outputs:
            raise ValueError(
                f"Example {example.id} has no outputs to use as a reference."
            )
        if self.reference_key is None:
            if len(example.outputs) > 1:
                raise ValueError(
                    f"Example {example.id} has multiple outputs, so you must"
                    " specify a reference_key."
                )
            else:
                return list(example.outputs.values())[0]
        elif self.reference_key not in example.outputs:
            raise ValueError(
                f"Example {example.id} does not have reference key"
                f" {self.reference_key}."
            )
        return {"reference": example.outputs[self.reference_key]}

    def __call__(self, example: Example) -> Dict[str, str]:
        """Maps the Run and Example to a dictionary."""
        return self.map(example)


class AgentTrajectoryRunEvaluatorChain(Chain, RunEvaluator):
    """Assumble the agent trajectory from a nested run and evaluate it."""

    run_mapper: AgentTrajectoryRunMapper = Field(
        default_factory=AgentTrajectoryRunMapper
    )
    """Maps the Run to a dictionary with 'prediction', 'input', 'agent_trajectory',
    and optionally 'reference' strings."""
    example_mapper: Optional[AgentTrajectoryExampleMapper] = None
    """Maps the Example (dataset row) to a dictionary
    with a 'reference' string."""
    name: str = Field(default="agent_trajectory")
    """The name of the evaluation metric."""
    agent_trajectory_evaluator: AgentTrajectoryEvaluator
    """The evaluation chain."""

    @property
    def input_keys(self) -> List[str]:
        return ["run", "example"]

    @property
    def output_keys(self) -> List[str]:
        return ["feedback"]

    def _prepare_input(self, inputs: Dict[str, Any]) -> AgentInput:
        run: Run = inputs["run"]
        example: Optional[Example] = inputs.get("example")
        evaluation_input = self.run_mapper(run)
        if example and self.example_mapper:
            reference_info = self.example_mapper(example)
            evaluation_input["reference"] = reference_info["reference"]
        elif self.agent_trajectory_evaluator.requires_reference:
            raise ValueError(
                f"Evaluator {self.name} requires an reference"
                " example from the dataset,"
                f" but none was provided for run {run.id}."
            )
        return evaluation_input

    def _prepare_output(self, output: Dict[str, Any]) -> EvaluationResult:
        evaluation_result = EvaluationResult(key=self.name, **output)
        if RUN_KEY in output:
            # TODO: Not currently surfaced. Update
            evaluation_result.evaluator_info[RUN_KEY] = output[RUN_KEY]
        return evaluation_result

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Call the evaluation chain."""
        evaluation_input = self._prepare_input(inputs)
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = self.agent_trajectory_evaluator.evaluate_agent_trajectory(
            prediction=evaluation_input["prediction"],
            agent_trajectory=evaluation_input["agent_trajectory"],
            input=evaluation_input["input"],
            reference=evaluation_input.get("reference"),
            callbacks=callbacks,
        )
        evaluation_result = self._prepare_output(chain_output)
        return {"feedback": evaluation_result}

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> Dict[str, Any]:
        """Call the evaluation chain."""
        evaluation_input = self._prepare_input(inputs)
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = await self.agent_trajectory_evaluator.aevaluate_agent_trajectory(
            **evaluation_input,
            callbacks=callbacks,
        )
        evaluation_result = self._prepare_output(chain_output)
        return {"feedback": evaluation_result}

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        """Evaluate an example."""
        return self({"run": run, "example": example})["feedback"]

    async def aevaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        """Evaluate an example."""
        result = await self.acall({"run": run, "example": example})
        return result["feedback"]

    @classmethod
    def from_model_and_evaluator(
        cls,
        model: Chain,
        evaluator: AgentTrajectoryEvaluator,
        input_key: str = "input",
        prediction_key: str = "output",
        reference_key: Optional[str] = None,
    ) -> AgentTrajectoryRunEvaluatorChain:
        """Create a AgentTrajectoryRunEvaluatorChain from a model and evaluator."""
        if not isinstance(model, Chain):
            raise NotImplementedError(
                f"{cls.__name__} currently only supports"
                " evaluating chains."
                f"Expected AgentExecutor. Got: {type(model)}"
            )
        if not isinstance(model, AgentExecutor):
            logger.warning("")
        run_mapper = AgentTrajectoryRunMapper.from_chain(
            model, input_key=input_key, prediction_key=prediction_key
        )
        if reference_key is not None:
            example_mapper = AgentTrajectoryExampleMapper(reference_key=reference_key)
        elif evaluator.requires_reference:
            # We could potentially auto-infer if there is only one string in the
            # example, but it's preferred to raise earlier.
            raise ValueError(
                f"Evaluator {evaluator.evaluation_name} requires a reference"
                " example from the dataset. Please specify the reference key from"
                " amongst the dataset outputs keys."
            )
        else:
            example_mapper = None
        return cls(
            name=evaluator.evaluation_name,
            run_mapper=run_mapper,
            example_mapper=example_mapper,
            agent_trajectory_evaluator=evaluator,
        )

"""Run evaluator wrapper for string evaluators."""

from __future__ import annotations

import logging
import uuid
from abc import abstractmethod
from typing import Any, cast

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.load.dump import dumpd
from langchain_core.load.load import load
from langchain_core.load.serializable import Serializable
from langchain_core.messages import BaseMessage, get_buffer_string, messages_from_dict
from langsmith import EvaluationResult, RunEvaluator
from langsmith.schemas import DataType, Example, Run
from typing_extensions import override

from langchain_classic.chains.base import Chain
from langchain_classic.evaluation.schema import StringEvaluator
from langchain_classic.schema import RUN_KEY

_logger = logging.getLogger(__name__)


def _get_messages_from_run_dict(messages: list[dict]) -> list[BaseMessage]:
    if not messages:
        return []
    first_message = messages[0]
    if "lc" in first_message:
        return [load(dumpd(message)) for message in messages]
    return messages_from_dict(messages)


class StringRunMapper(Serializable):
    """Extract items to evaluate from the run object."""

    @property
    def output_keys(self) -> list[str]:
        """The keys to extract from the run."""
        return ["prediction", "input"]

    @abstractmethod
    def map(self, run: Run) -> dict[str, str]:
        """Maps the Run to a dictionary."""

    def __call__(self, run: Run) -> dict[str, str]:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            msg = f"Run {run.id} has no outputs to evaluate."
            raise ValueError(msg)
        return self.map(run)


class LLMStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object."""

    def serialize_chat_messages(self, messages: list[dict] | list[list[dict]]) -> str:
        """Extract the input messages from the run."""
        if isinstance(messages, list) and messages:
            if isinstance(messages[0], dict):
                chat_messages = _get_messages_from_run_dict(
                    cast("list[dict]", messages)
                )
            elif isinstance(messages[0], list):
                # Runs from Tracer have messages as a list of lists of dicts
                chat_messages = _get_messages_from_run_dict(messages[0])
            else:
                msg = f"Could not extract messages to evaluate {messages}"  # type: ignore[unreachable]
                raise ValueError(msg)
            return get_buffer_string(chat_messages)
        msg = f"Could not extract messages to evaluate {messages}"
        raise ValueError(msg)

    def serialize_inputs(self, inputs: dict) -> str:
        """Serialize inputs.

        Args:
            inputs: The inputs from the run, expected to contain prompts or messages.

        Returns:
            The serialized input text from the prompts or messages.

        Raises:
            ValueError: If neither prompts nor messages are found in the inputs.
        """
        if "prompts" in inputs:  # Should we even accept this?
            input_ = "\n\n".join(inputs["prompts"])
        elif "prompt" in inputs:
            input_ = inputs["prompt"]
        elif "messages" in inputs:
            input_ = self.serialize_chat_messages(inputs["messages"])
        else:
            msg = "LLM Run must have either messages or prompts as inputs."
            raise ValueError(msg)
        return input_

    def serialize_outputs(self, outputs: dict) -> str:
        """Serialize outputs.

        Args:
            outputs: The outputs from the run, expected to contain generations.

        Returns:
            The serialized output text from the first generation.

        Raises:
            ValueError: If no generations are found in the outputs,
            or if the generations are empty.
        """
        if not outputs.get("generations"):
            msg = "Cannot evaluate LLM Run without generations."
            raise ValueError(msg)
        generations: list[dict] | list[list[dict]] = outputs["generations"]
        if not generations:
            msg = "Cannot evaluate LLM run with empty generations."
            raise ValueError(msg)
        first_generation: dict | list[dict] = generations[0]
        if isinstance(first_generation, list):
            # Runs from Tracer have generations as a list of lists of dicts
            # Whereas Runs from the API have a list of dicts
            first_generation = first_generation[0]
        if "message" in first_generation:
            output_ = self.serialize_chat_messages([first_generation["message"]])
        else:
            output_ = first_generation["text"]
        return output_

    def map(self, run: Run) -> dict[str, str]:
        """Maps the Run to a dictionary."""
        if run.run_type != "llm":
            msg = "LLM RunMapper only supports LLM runs."
            raise ValueError(msg)
        if not run.outputs:
            if run.error:
                msg = f"Cannot evaluate errored LLM run {run.id}: {run.error}"
                raise ValueError(msg)
            msg = f"Run {run.id} has no outputs. Cannot evaluate this run."
            raise ValueError(msg)
        try:
            inputs = self.serialize_inputs(run.inputs)
        except Exception as e:
            msg = f"Could not parse LM input from run inputs {run.inputs}"
            raise ValueError(msg) from e
        try:
            output_ = self.serialize_outputs(run.outputs)
        except Exception as e:
            msg = f"Could not parse LM prediction from run outputs {run.outputs}"
            raise ValueError(msg) from e
        return {"input": inputs, "prediction": output_}


class ChainStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object from a chain."""

    input_key: str | None = None
    """The key from the model Run's inputs to use as the eval input.
    If not provided, will use the only input key or raise an
    error if there are multiple."""
    prediction_key: str | None = None
    """The key from the model Run's outputs to use as the eval prediction.
    If not provided, will use the only output key or raise an error
    if there are multiple."""

    def _get_key(self, source: dict, key: str | None, which: str) -> str:
        if key is not None:
            return source[key]
        if len(source) == 1:
            return next(iter(source.values()))
        msg = (
            f"Could not map run {which} with multiple keys: "
            f"{source}\nPlease manually specify a {which}_key"
        )
        raise ValueError(msg)

    def map(self, run: Run) -> dict[str, str]:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            msg = (
                f"Run with ID {run.id} lacks outputs required for evaluation."
                " Ensure the Run has valid outputs."
            )
            raise ValueError(msg)
        if self.input_key is not None and self.input_key not in run.inputs:
            msg = (
                f"Run with ID {run.id} is missing the expected input key"
                f" '{self.input_key}'.\nAvailable input keys in this Run"
                f"  are: {run.inputs.keys()}.\nAdjust the evaluator's"
                f" input_key or ensure your input data includes key"
                f" '{self.input_key}'."
            )
            raise ValueError(msg)
        if self.prediction_key is not None and self.prediction_key not in run.outputs:
            available_keys = ", ".join(run.outputs.keys())
            msg = (
                f"Run with ID {run.id} doesn't have the expected prediction key"
                f" '{self.prediction_key}'. Available prediction keys in this Run are:"
                f" {available_keys}. Adjust the evaluator's prediction_key or"
                " ensure the Run object's outputs the expected key."
            )
            raise ValueError(msg)

        input_ = self._get_key(run.inputs, self.input_key, "input")
        prediction = self._get_key(run.outputs, self.prediction_key, "prediction")
        return {
            "input": input_,
            "prediction": prediction,
        }


class ToolStringRunMapper(StringRunMapper):
    """Map an input to the tool."""

    @override
    def map(self, run: Run) -> dict[str, str]:
        if not run.outputs:
            msg = f"Run {run.id} has no outputs to evaluate."
            raise ValueError(msg)
        return {"input": run.inputs["input"], "prediction": run.outputs["output"]}


class StringExampleMapper(Serializable):
    """Map an example, or row in the dataset, to the inputs of an evaluation."""

    reference_key: str | None = None

    @property
    def output_keys(self) -> list[str]:
        """The keys to extract from the run."""
        return ["reference"]

    def serialize_chat_messages(self, messages: list[dict]) -> str:
        """Extract the input messages from the run."""
        chat_messages = _get_messages_from_run_dict(messages)
        return get_buffer_string(chat_messages)

    def map(self, example: Example) -> dict[str, str]:
        """Maps the Example, or dataset row to a dictionary."""
        if not example.outputs:
            msg = f"Example {example.id} has no outputs to use as a reference."
            raise ValueError(msg)
        if self.reference_key is None:
            if len(example.outputs) > 1:
                msg = (
                    f"Example {example.id} has multiple outputs, so you must"
                    " specify a reference_key."
                )
                raise ValueError(msg)
            output = next(iter(example.outputs.values()))
        elif self.reference_key not in example.outputs:
            msg = (
                f"Example {example.id} does not have reference key"
                f" {self.reference_key}."
            )
            raise ValueError(msg)
        else:
            output = example.outputs[self.reference_key]
        return {
            "reference": self.serialize_chat_messages([output])
            if isinstance(output, dict) and output.get("type") and output.get("data")
            else output,
        }

    def __call__(self, example: Example) -> dict[str, str]:
        """Maps the Run and Example to a dictionary."""
        if not example.outputs:
            msg = f"Example {example.id} has no outputs to use as areference label."
            raise ValueError(msg)
        return self.map(example)


class StringRunEvaluatorChain(Chain, RunEvaluator):
    """Evaluate Run and optional examples."""

    run_mapper: StringRunMapper
    """Maps the Run to a dictionary with 'input' and 'prediction' strings."""
    example_mapper: StringExampleMapper | None = None
    """Maps the Example (dataset row) to a dictionary
    with a 'reference' string."""
    name: str
    """The name of the evaluation metric."""
    string_evaluator: StringEvaluator
    """The evaluation chain."""

    @property
    @override
    def input_keys(self) -> list[str]:
        return ["run", "example"]

    @property
    @override
    def output_keys(self) -> list[str]:
        return ["feedback"]

    def _prepare_input(self, inputs: dict[str, Any]) -> dict[str, str]:
        run: Run = inputs["run"]
        example: Example | None = inputs.get("example")
        evaluate_strings_inputs = self.run_mapper(run)
        if not self.string_evaluator.requires_input:
            # Hide warning about unused input
            evaluate_strings_inputs.pop("input", None)
        if example and self.example_mapper and self.string_evaluator.requires_reference:
            evaluate_strings_inputs.update(self.example_mapper(example))
        elif self.string_evaluator.requires_reference:
            msg = (
                f"Evaluator {self.name} requires an reference"
                " example from the dataset,"
                f" but none was provided for run {run.id}."
            )
            raise ValueError(msg)
        return evaluate_strings_inputs

    def _prepare_output(self, output: dict[str, Any]) -> dict[str, Any]:
        evaluation_result = EvaluationResult(
            key=self.name,
            comment=output.get("reasoning"),
            **output,
        )
        if RUN_KEY in output:
            # TODO: Not currently surfaced. Update
            evaluation_result.evaluator_info[RUN_KEY] = output[RUN_KEY]
        return {"feedback": evaluation_result}

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Call the evaluation chain."""
        evaluate_strings_inputs = self._prepare_input(inputs)
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = self.string_evaluator.evaluate_strings(
            **evaluate_strings_inputs,
            callbacks=callbacks,
            include_run_info=True,
        )
        return self._prepare_output(chain_output)

    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        """Call the evaluation chain."""
        evaluate_strings_inputs = self._prepare_input(inputs)
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        chain_output = await self.string_evaluator.aevaluate_strings(
            **evaluate_strings_inputs,
            callbacks=callbacks,
            include_run_info=True,
        )
        return self._prepare_output(chain_output)

    def _prepare_evaluator_output(self, output: dict[str, Any]) -> EvaluationResult:
        feedback: EvaluationResult = output["feedback"]
        if RUN_KEY not in feedback.evaluator_info:
            feedback.evaluator_info[RUN_KEY] = output[RUN_KEY]
        return feedback

    @override
    def evaluate_run(
        self,
        run: Run,
        example: Example | None = None,
        evaluator_run_id: uuid.UUID | None = None,
    ) -> EvaluationResult:
        """Evaluate an example."""
        try:
            result = self({"run": run, "example": example}, include_run_info=True)
            return self._prepare_evaluator_output(result)
        except Exception as e:
            _logger.exception("Error evaluating run %s", run.id)
            return EvaluationResult(
                key=self.string_evaluator.evaluation_name,
                comment=f"Error evaluating run {run.id}: {e}",
                # TODO: Add run ID once we can declare it via callbacks
            )

    @override
    async def aevaluate_run(
        self,
        run: Run,
        example: Example | None = None,
        evaluator_run_id: uuid.UUID | None = None,
    ) -> EvaluationResult:
        """Evaluate an example."""
        try:
            result = await self.acall(
                {"run": run, "example": example},
                include_run_info=True,
            )
            return self._prepare_evaluator_output(result)
        except Exception as e:
            _logger.exception("Error evaluating run %s", run.id)
            return EvaluationResult(
                key=self.string_evaluator.evaluation_name,
                comment=f"Error evaluating run {run.id}: {e}",
            )

    @classmethod
    def from_run_and_data_type(
        cls,
        evaluator: StringEvaluator,
        run_type: str,
        data_type: DataType,
        input_key: str | None = None,
        prediction_key: str | None = None,
        reference_key: str | None = None,
        tags: list[str] | None = None,
    ) -> StringRunEvaluatorChain:
        """Create a StringRunEvaluatorChain.

        Create a StringRunEvaluatorChain from an evaluator and the run and dataset
        types.

        This method provides an easy way to instantiate a StringRunEvaluatorChain, by
        taking an evaluator and information about the type of run and the data.
        The method supports LLM and chain runs.

        Args:
            evaluator: The string evaluator to use.
            run_type: The type of run being evaluated.
                Supported types are LLM and Chain.
            data_type: The type of dataset used in the run.
            input_key: The key used to map the input from the run.
            prediction_key: The key used to map the prediction from the run.
            reference_key: The key used to map the reference from the dataset.
            tags: List of tags to attach to the evaluation chain.

        Returns:
            The instantiated evaluation chain.

        Raises:
            If the run type is not supported, or if the evaluator requires a
            reference from the dataset but the reference key is not provided.

        """
        # Configure how run inputs/predictions are passed to the evaluator
        if run_type == "llm":
            run_mapper: StringRunMapper = LLMStringRunMapper()
        elif run_type == "chain":
            run_mapper = ChainStringRunMapper(
                input_key=input_key,
                prediction_key=prediction_key,
            )
        else:
            msg = f"Unsupported run type {run_type}. Expected one of 'llm' or 'chain'."
            raise ValueError(msg)

        # Configure how example rows are fed as a reference string to the evaluator
        if (
            reference_key is not None
            or data_type in (DataType.llm, DataType.chat)
            or evaluator.requires_reference
        ):
            example_mapper = StringExampleMapper(reference_key=reference_key)
        elif evaluator.requires_reference:
            msg = (  # type: ignore[unreachable]
                f"Evaluator {evaluator.evaluation_name} requires a reference"
                " example from the dataset. Please specify the reference key from"
                " amongst the dataset outputs keys."
            )
            raise ValueError(msg)
        else:
            example_mapper = None
        return cls(
            name=evaluator.evaluation_name,
            run_mapper=run_mapper,
            example_mapper=example_mapper,
            string_evaluator=evaluator,
            tags=tags,
        )

"""Run evaluator wrapper for string evaluators."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

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

from langchain.chains.base import Chain
from langchain.evaluation.schema import StringEvaluator
from langchain.schema import RUN_KEY


def _get_messages_from_run_dict(messages: List[dict]) -> List[BaseMessage]:
    if not messages:
        return []
    first_message = messages[0]
    if "lc" in first_message:
        return [load(dumpd(message)) for message in messages]
    else:
        return messages_from_dict(messages)


class StringRunMapper(Serializable):
    """Extract items to evaluate from the run object."""

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ["prediction", "input"]

    @abstractmethod
    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""

    def __call__(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            raise ValueError(f"Run {run.id} has no outputs to evaluate.")
        return self.map(run)


class LLMStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object."""

    def serialize_chat_messages(self, messages: List[Dict]) -> str:
        """Extract the input messages from the run."""
        if isinstance(messages, list) and messages:
            if isinstance(messages[0], dict):
                chat_messages = _get_messages_from_run_dict(messages)
            elif isinstance(messages[0], list):
                # Runs from Tracer have messages as a list of lists of dicts
                chat_messages = _get_messages_from_run_dict(messages[0])
            else:
                raise ValueError(f"Could not extract messages to evaluate {messages}")
            return get_buffer_string(chat_messages)
        raise ValueError(f"Could not extract messages to evaluate {messages}")

    def serialize_inputs(self, inputs: Dict) -> str:
        if "prompts" in inputs:  # Should we even accept this?
            input_ = "\n\n".join(inputs["prompts"])
        elif "prompt" in inputs:
            input_ = inputs["prompt"]
        elif "messages" in inputs:
            input_ = self.serialize_chat_messages(inputs["messages"])
        else:
            raise ValueError("LLM Run must have either messages or prompts as inputs.")
        return input_

    def serialize_outputs(self, outputs: Dict) -> str:
        if not outputs.get("generations"):
            raise ValueError("Cannot evaluate LLM Run without generations.")
        generations: List[Dict] = outputs["generations"]
        if not generations:
            raise ValueError("Cannot evaluate LLM run with empty generations.")
        first_generation: Dict = generations[0]
        if isinstance(first_generation, list):
            # Runs from Tracer have generations as a list of lists of dicts
            # Whereas Runs from the API have a list of dicts
            first_generation = first_generation[0]
        if "message" in first_generation:
            output_ = self.serialize_chat_messages([first_generation["message"]])
        else:
            output_ = first_generation["text"]
        return output_

    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if run.run_type != "llm":
            raise ValueError("LLM RunMapper only supports LLM runs.")
        elif not run.outputs:
            if run.error:
                raise ValueError(
                    f"Cannot evaluate errored LLM run {run.id}: {run.error}"
                )
            else:
                raise ValueError(
                    f"Run {run.id} has no outputs. Cannot evaluate this run."
                )
        else:
            try:
                inputs = self.serialize_inputs(run.inputs)
            except Exception as e:
                raise ValueError(
                    f"Could not parse LM input from run inputs {run.inputs}"
                ) from e
            try:
                output_ = self.serialize_outputs(run.outputs)
            except Exception as e:
                raise ValueError(
                    f"Could not parse LM prediction from run outputs {run.outputs}"
                ) from e
            return {"input": inputs, "prediction": output_}


class ChainStringRunMapper(StringRunMapper):
    """Extract items to evaluate from the run object from a chain."""

    input_key: Optional[str] = None
    """The key from the model Run's inputs to use as the eval input.
    If not provided, will use the only input key or raise an
    error if there are multiple."""
    prediction_key: Optional[str] = None
    """The key from the model Run's outputs to use as the eval prediction.
    If not provided, will use the only output key or raise an error
    if there are multiple."""

    def _get_key(self, source: Dict, key: Optional[str], which: str) -> str:
        if key is not None:
            return source[key]
        elif len(source) == 1:
            return next(iter(source.values()))
        else:
            raise ValueError(
                f"Could not map run {which} with multiple keys: "
                f"{source}\nPlease manually specify a {which}_key"
            )

    def map(self, run: Run) -> Dict[str, str]:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            raise ValueError(
                f"Run with ID {run.id} lacks outputs required for evaluation."
                " Ensure the Run has valid outputs."
            )
        if self.input_key is not None and self.input_key not in run.inputs:
            raise ValueError(
                f"Run with ID {run.id} is missing the expected input key"
                f" '{self.input_key}'.\nAvailable input keys in this Run"
                f"  are: {run.inputs.keys()}.\nAdjust the evaluator's"
                f" input_key or ensure your input data includes key"
                f" '{self.input_key}'."
            )
        elif self.prediction_key is not None and self.prediction_key not in run.outputs:
            available_keys = ", ".join(run.outputs.keys())
            raise ValueError(
                f"Run with ID {run.id} doesn't have the expected prediction key"
                f" '{self.prediction_key}'. Available prediction keys in this Run are:"
                f" {available_keys}. Adjust the evaluator's prediction_key or"
                " ensure the Run object's outputs the expected key."
            )

        else:
            input_ = self._get_key(run.inputs, self.input_key, "input")
            prediction = self._get_key(run.outputs, self.prediction_key, "prediction")
            return {
                "input": input_,
                "prediction": prediction,
            }


class ToolStringRunMapper(StringRunMapper):
    """Map an input to the tool."""

    def map(self, run: Run) -> Dict[str, str]:
        if not run.outputs:
            raise ValueError(f"Run {run.id} has no outputs to evaluate.")
        return {"input": run.inputs["input"], "prediction": run.outputs["output"]}


class StringExampleMapper(Serializable):
    """Map an example, or row in the dataset, to the inputs of an evaluation."""

    reference_key: Optional[str] = None

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ["reference"]

    def serialize_chat_messages(self, messages: List[Dict]) -> str:
        """Extract the input messages from the run."""
        chat_messages = _get_messages_from_run_dict(messages)
        return get_buffer_string(chat_messages)

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
                output = list(example.outputs.values())[0]
        elif self.reference_key not in example.outputs:
            raise ValueError(
                f"Example {example.id} does not have reference key"
                f" {self.reference_key}."
            )
        else:
            output = example.outputs[self.reference_key]
        return {
            "reference": self.serialize_chat_messages([output])
            if isinstance(output, dict) and output.get("type") and output.get("data")
            else output
        }

    def __call__(self, example: Example) -> Dict[str, str]:
        """Maps the Run and Example to a dictionary."""
        if not example.outputs:
            raise ValueError(
                f"Example {example.id} has no outputs to use as areference label."
            )
        return self.map(example)


class StringRunEvaluatorChain(Chain, RunEvaluator):
    """Evaluate Run and optional examples."""

    run_mapper: StringRunMapper
    """Maps the Run to a dictionary with 'input' and 'prediction' strings."""
    example_mapper: Optional[StringExampleMapper] = None
    """Maps the Example (dataset row) to a dictionary
    with a 'reference' string."""
    name: str
    """The name of the evaluation metric."""
    string_evaluator: StringEvaluator
    """The evaluation chain."""

    @property
    def input_keys(self) -> List[str]:
        return ["run", "example"]

    @property
    def output_keys(self) -> List[str]:
        return ["feedback"]

    def _prepare_input(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        run: Run = inputs["run"]
        example: Optional[Example] = inputs.get("example")
        evaluate_strings_inputs = self.run_mapper(run)
        if not self.string_evaluator.requires_input:
            # Hide warning about unused input
            evaluate_strings_inputs.pop("input", None)
        if example and self.example_mapper and self.string_evaluator.requires_reference:
            evaluate_strings_inputs.update(self.example_mapper(example))
        elif self.string_evaluator.requires_reference:
            raise ValueError(
                f"Evaluator {self.name} requires an reference"
                " example from the dataset,"
                f" but none was provided for run {run.id}."
            )
        return evaluate_strings_inputs

    def _prepare_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        evaluation_result = EvaluationResult(
            key=self.name, comment=output.get("reasoning"), **output
        )
        if RUN_KEY in output:
            # TODO: Not currently surfaced. Update
            evaluation_result.evaluator_info[RUN_KEY] = output[RUN_KEY]
        return {"feedback": evaluation_result}

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
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
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
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

    def _prepare_evaluator_output(self, output: Dict[str, Any]) -> EvaluationResult:
        feedback: EvaluationResult = output["feedback"]
        if RUN_KEY not in feedback.evaluator_info:
            feedback.evaluator_info[RUN_KEY] = output[RUN_KEY]
        return feedback

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        """Evaluate an example."""
        try:
            result = self({"run": run, "example": example}, include_run_info=True)
            return self._prepare_evaluator_output(result)
        except Exception as e:
            return EvaluationResult(
                key=self.string_evaluator.evaluation_name,
                comment=f"Error evaluating run {run.id}: {e}",
                # TODO: Add run ID once we can declare it via callbacks
            )

    async def aevaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        """Evaluate an example."""
        try:
            result = await self.acall(
                {"run": run, "example": example}, include_run_info=True
            )
            return self._prepare_evaluator_output(result)
        except Exception as e:
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
        input_key: Optional[str] = None,
        prediction_key: Optional[str] = None,
        reference_key: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> StringRunEvaluatorChain:
        """
        Create a StringRunEvaluatorChain from an evaluator and the run and dataset types.

        This method provides an easy way to instantiate a StringRunEvaluatorChain, by
        taking an evaluator and information about the type of run and the data.
        The method supports LLM and chain runs.

        Args:
            evaluator (StringEvaluator): The string evaluator to use.
            run_type (str): The type of run being evaluated.
                Supported types are LLM and Chain.
            data_type (DataType): The type of dataset used in the run.
            input_key (str, optional): The key used to map the input from the run.
            prediction_key (str, optional): The key used to map the prediction from the run.
            reference_key (str, optional): The key used to map the reference from the dataset.
            tags (List[str], optional): List of tags to attach to the evaluation chain.

        Returns:
            StringRunEvaluatorChain: The instantiated evaluation chain.

        Raises:
            ValueError: If the run type is not supported, or if the evaluator requires a
                reference from the dataset but the reference key is not provided.

        """  # noqa: E501

        # Configure how run inputs/predictions are passed to the evaluator
        if run_type == "llm":
            run_mapper: StringRunMapper = LLMStringRunMapper()
        elif run_type == "chain":
            run_mapper = ChainStringRunMapper(
                input_key=input_key, prediction_key=prediction_key
            )
        else:
            raise ValueError(
                f"Unsupported run type {run_type}. Expected one of 'llm' or 'chain'."
            )

        # Configure how example rows are fed as a reference string to the evaluator
        if (
            reference_key is not None
            or data_type in (DataType.llm, DataType.chat)
            or evaluator.requires_reference
        ):
            example_mapper = StringExampleMapper(reference_key=reference_key)
        elif evaluator.requires_reference:
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
            string_evaluator=evaluator,
            tags=tags,
        )

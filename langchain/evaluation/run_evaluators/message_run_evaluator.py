"""Run evaluator mapper for message evaluators."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, TypedDict
from langchain.schema import BaseMessage

from langchainplus_sdk.schemas import Example, Run
from langchain.load.serializable import Serializable
from langchain.schema import messages_from_dict


class RunMapping(TypedDict):
    prediction: BaseMessage
    input: List[BaseMessage]


class ExampleMapping(TypedDict):
    reference: BaseMessage


class MessageRunMapper(Serializable):
    """Extract items to evaluate from run object"""

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ["prediction", "input"]

    @abstractmethod
    def map(self, run: Run) -> RunMapping:
        """Maps the Run to a dictionary."""

    def __call__(self, run: Run) -> RunMapping:
        """Maps the Run to a dictionary."""
        if not run.outputs:
            raise ValueError(f"Run {run.id} has no outputs to evaluate.")
        return self.map(run)


class MessageExampleMapper(Serializable):
    """Map an example, or row in the dataset, to the inputs of an evaluation."""

    reference_key: Optional[str] = None

    @property
    def output_keys(self) -> List[str]:
        """The keys to extract from the run."""
        return ["reference"]

    def map(self, example: Example) -> ExampleMapping:
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
                return {
                    "reference": output if isinstance(output, BaseMessage) else messages_from_dict([output])[0]
                }
        elif self.reference_key not in example.outputs:
            raise ValueError(
                f"Example {example.id} does not have reference key"
                f" {self.reference_key}."
            )
        output = example.outputs[self.reference_key]
        return {"reference": output if isinstance(output, BaseMessage) else messages_from_dict([output])[0]}

    def __call__(self, example: Example) -> ExampleMapping:
        """Maps the Run and Example to a dictionary."""
        if not example.outputs:
            raise ValueError(
                f"Example {example.id} has no outputs to use as areference label."
            )
        return self.map(example)


class ChatModelMessageRunMapper(MessageRunMapper):
    """Extract items to evaluate from run object."""

    @staticmethod
    def extract_inputs(inputs: Dict) -> List[BaseMessage]:
        if not inputs.get("messages"):
            raise ValueError("Run must have messages as inputs.")
        if "messages" in inputs:
            if isinstance(inputs["messages"], list) and inputs["messages"]:
                if isinstance(inputs["messages"][0], BaseMessage):
                    return messages_from_dict(inputs["messages"])
                elif isinstance(inputs["messages"][0], list):
                    # Runs from Tracer have messages as a list of lists of dicts
                    return messages_from_dict(inputs["messages"][0])
        raise ValueError(f"Could not extract messages from inputs: {inputs}")

    @staticmethod
    def extract_outputs(outputs: Dict) -> BaseMessage:
        if not outputs.get("generations"):
            raise ValueError("LLM Run must have generations as outputs.")
        first_generation: Dict = outputs["generations"][0]
        if isinstance(first_generation, list):
            # Runs from Tracer have generations as a list of lists of dicts
            # Whereas Runs from the API have a list of dicts
            first_generation = first_generation[0]
        if "message" in first_generation:
            return messages_from_dict([first_generation["message"]])[0]

    def map(self, run: Run) -> RunMapping:
        """Maps the Run to a dictionary."""
        if run.run_type != "llm":
            raise ValueError("ChatModel RunMapper only supports LangSmith runs of type llm.")
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
                inputs = self.extract_inputs(run.inputs)
            except Exception as e:
                raise ValueError(
                    f"Could not parse LM input from run inputs {run.inputs}"
                ) from e
            try:
                output_ = self.extract_outputs(run.outputs)
            except Exception as e:
                raise ValueError(
                    f"Could not parse LM prediction from run outputs {run.outputs}"
                ) from e
            return {"input": inputs, "prediction": output_}
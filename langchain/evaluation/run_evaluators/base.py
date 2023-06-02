from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, TypeVar

from langchainplus_sdk.evaluation.evaluator import EvaluationResult
from langchainplus_sdk.schemas import Example, Run
from pydantic import BaseModel
from pyparsing import abstractmethod
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain

from langchain.chains.llm import LLMChain
from langchain.schema import BaseOutputParser


class RunEvalInputMapper:
    """Map the inputs of a run to the inputs of an evaluation."""

    @abstractmethod
    def map(self, run: Run, example: Optional[Example] = None) -> Dict[str, str]:
        """Maps the Run and Optional[Example] to a dictionary"""


class StringRunEvalInputMapper(RunEvalInputMapper, BaseModel):
    """Maps the Run and Optional[Example] to a dictionary."""

    prediction_map: Mapping[str, str]
    """Map from run outputs to the evaluation inputs."""
    input_map: Mapping[str, str]
    """Map from run inputs to the evaluation inputs."""
    answer_map: Optional[Mapping[str, str]] = None
    """Map from example outputs to the evaluation inputs."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def map(self, run: Run, example: Optional[Example] = None) -> Dict[str, str]:
        """Maps the Run and Optional[Example] to a dictionary"""
        if run.outputs is None:
            raise ValueError("Run outputs cannot be None.")

        data = {
            value: run.outputs.get(key) for key, value in self.prediction_map.items()
        }
        data.update(
            {value: run.inputs.get(key) for key, value in self.input_map.items()}
        )
        if self.answer_map and example and example.outputs:
            data.update(
                {
                    value: example.outputs.get(key)
                    for key, value in self.answer_map.items()
                }
            )
        return data


class RunEvaluatorOutputParser(BaseOutputParser[EvaluationResult]):
    """Parse the output of a run."""


class ChoicesOutputParser(RunEvaluatorOutputParser):
    """Parse a feedback run with optional choices."""

    evaluation_name: str
    choices_map: Optional[Dict[str, int]] = None

    def parse(self, text: str) -> EvaluationResult:
        """Parse the last line of the text and return an evaluation result."""
        lines = text.strip().split()
        value = lines[-1]
        score = self.choices_map.get(value, 0) if self.choices_map else None
        comment = " ".join(lines[:-1]) if len(lines) > 1 else None
        return EvaluationResult(
            key=self.evaluation_name,
            score=score,
            value=value,
            comment=comment,
        )


T = TypeVar("T", bound="RunEvaluator")


class RunEvaluator(Chain):
    """Evaluate Run and optional examples."""

    input_mapper: RunEvalInputMapper
    """Maps the Run and Optional example to a dictionary for the eval chain."""
    eval_chain: LLMChain
    """The evaluation chain."""
    output_parser: RunEvaluatorOutputParser
    """Parse the output of the eval chain into feedback."""

    @property
    def input_keys(self) -> List[str]:
        return ["run", "example"]

    @property
    def output_keys(self) -> List[str]:
        return ["feedback"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Call the evaluation chain."""
        run: Run = inputs["run"]
        example: Optional[Example] = inputs.get("example")
        chain_input = self.input_mapper.map(run, example)
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        chain_output = self.eval_chain(chain_input, callbacks=_run_manager.get_child())
        feedback = chain_output["text"]
        return {"feedback": self.output_parser.parse(feedback)}

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        """Evaluate an example."""
        return self({"run": run, "example": example})["feedback"]

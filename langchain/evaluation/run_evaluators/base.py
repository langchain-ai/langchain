from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from langchainplus_sdk import EvaluationResult, RunEvaluator
from langchainplus_sdk.schemas import Example, Run

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.schema import BaseOutputParser


class RunEvalInputMapper:
    """Map the inputs of a run to the inputs of an evaluation."""

    @abstractmethod
    def map(self, run: Run, example: Optional[Example] = None) -> Dict[str, Any]:
        """Maps the Run and Optional[Example] to a dictionary"""


class RunEvaluatorOutputParser(BaseOutputParser[EvaluationResult]):
    """Parse the output of a run."""

    eval_chain_output_key: str = "text"

    def parse_chain_output(self, output: Dict[str, Any]) -> EvaluationResult:
        """Parse the output of a run."""
        text = output[self.eval_chain_output_key]
        return self.parse(text)


class RunEvaluatorChain(Chain, RunEvaluator):
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
        feedback = self.output_parser.parse_chain_output(chain_output)
        return {"feedback": feedback}

    def evaluate_run(
        self, run: Run, example: Optional[Example] = None
    ) -> EvaluationResult:
        """Evaluate an example."""
        return self({"run": run, "example": example})["feedback"]

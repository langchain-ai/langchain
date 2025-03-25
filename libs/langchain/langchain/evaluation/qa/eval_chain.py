"""LLM Chains for evaluating question answering."""

from __future__ import annotations

import re
import string
from typing import Any, List, Optional, Sequence, Tuple

from langchain_core.callbacks.manager import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from pydantic import ConfigDict

from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.eval_prompt import CONTEXT_PROMPT, COT_PROMPT, PROMPT
from langchain.evaluation.schema import LLMEvalChain, StringEvaluator
from langchain.schema import RUN_KEY


def _get_score(text: str) -> Optional[Tuple[str, int]]:
    match = re.search(r"grade:\s*(correct|incorrect)", text.strip(), re.IGNORECASE)
    if match:
        if match.group(1).upper() == "CORRECT":
            return "CORRECT", 1
        elif match.group(1).upper() == "INCORRECT":
            return "INCORRECT", 0
    try:
        first_word = (
            text.strip().split()[0].translate(str.maketrans("", "", string.punctuation))
        )
        if first_word.upper() == "CORRECT":
            return "CORRECT", 1
        elif first_word.upper() == "INCORRECT":
            return "INCORRECT", 0
        last_word = (
            text.strip()
            .split()[-1]
            .translate(str.maketrans("", "", string.punctuation))
        )
        if last_word.upper() == "CORRECT":
            return "CORRECT", 1
        elif last_word.upper() == "INCORRECT":
            return "INCORRECT", 0
    except IndexError:
        pass
    return None


def _parse_string_eval_output(text: str) -> dict:
    """Parse the output text.

    Args:
        text (str): The output text to parse.

    Returns:
        Any: The parsed output.
    """
    reasoning = text.strip()
    parsed_scores = _get_score(reasoning)
    if parsed_scores is None:
        value, score = None, None
    else:
        value, score = parsed_scores
    return {
        "reasoning": reasoning,
        "value": value,
        "score": score,
    }


class QAEvalChain(LLMChain, StringEvaluator, LLMEvalChain):
    """LLM Chain for evaluating question answering."""

    output_key: str = "results"  #: :meta private:

    model_config = ConfigDict(
        extra="ignore",
    )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return "correctness"

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def requires_input(self) -> bool:
        return True

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> QAEvalChain:
        """Load QA Eval Chain from LLM.

        Args:
            llm (BaseLanguageModel): the base language model to use.

            prompt (PromptTemplate): A prompt template containing the input_variables:
            'input', 'answer' and 'result' that will be used as the prompt
            for evaluation.
            Defaults to PROMPT.

            **kwargs: additional keyword arguments.

        Returns:
            QAEvalChain: the loaded QA eval chain.
        """
        prompt = prompt or PROMPT
        expected_input_vars = {"query", "answer", "result"}
        if expected_input_vars != set(prompt.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt.input_variables}"
            )
        return cls(llm=llm, prompt=prompt, **kwargs)

    def evaluate(
        self,
        examples: Sequence[dict],
        predictions: Sequence[dict],
        question_key: str = "query",
        answer_key: str = "answer",
        prediction_key: str = "result",
        *,
        callbacks: Callbacks = None,
    ) -> List[dict]:
        """Evaluate question answering examples and predictions."""
        inputs = [
            {
                "query": example[question_key],
                "answer": example[answer_key],
                "result": predictions[i][prediction_key],
            }
            for i, example in enumerate(examples)
        ]

        return self.apply(inputs, callbacks=callbacks)

    def _prepare_output(self, result: dict) -> dict:
        parsed_result = _parse_string_eval_output(result[self.output_key])
        if RUN_KEY in result:
            parsed_result[RUN_KEY] = result[RUN_KEY]
        return parsed_result

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): the LLM or chain prediction to evaluate.
            reference (Optional[str], optional): the reference label
                to evaluate against.
            input (Optional[str], optional): the input to consider during evaluation
            callbacks (Callbacks, optional): the callbacks to use for tracing.
            include_run_info (bool, optional): whether to include run info in the
                returned results.
            **kwargs: additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """
        result = self(
            {
                "query": input,
                "answer": reference,
                "result": prediction,
            },
            callbacks=callbacks,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    async def _aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        result = await self.acall(
            inputs={"query": input, "answer": reference, "result": prediction},
            callbacks=callbacks,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class ContextQAEvalChain(LLMChain, StringEvaluator, LLMEvalChain):
    """LLM Chain for evaluating QA w/o GT based on context"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        """Whether the chain requires a reference string."""
        return True

    @property
    def requires_input(self) -> bool:
        """Whether the chain requires an input string."""
        return True

    model_config = ConfigDict(
        extra="ignore",
    )

    @classmethod
    def _validate_input_vars(cls, prompt: PromptTemplate) -> None:
        expected_input_vars = {"query", "context", "result"}
        if expected_input_vars != set(prompt.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt.input_variables}"
            )

    @property
    def evaluation_name(self) -> str:
        return "Contextual Accuracy"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> ContextQAEvalChain:
        """Load QA Eval Chain from LLM.

        Args:
            llm (BaseLanguageModel): the base language model to use.

            prompt (PromptTemplate): A prompt template containing the input_variables:
            'query', 'context' and 'result' that will be used as the prompt
            for evaluation.
            Defaults to PROMPT.

            **kwargs: additional keyword arguments.

        Returns:
            ContextQAEvalChain: the loaded QA eval chain.
        """
        prompt = prompt or CONTEXT_PROMPT
        cls._validate_input_vars(prompt)
        return cls(llm=llm, prompt=prompt, **kwargs)

    def evaluate(
        self,
        examples: List[dict],
        predictions: List[dict],
        question_key: str = "query",
        context_key: str = "context",
        prediction_key: str = "result",
        *,
        callbacks: Callbacks = None,
    ) -> List[dict]:
        """Evaluate question answering examples and predictions."""
        inputs = [
            {
                "query": example[question_key],
                "context": example[context_key],
                "result": predictions[i][prediction_key],
            }
            for i, example in enumerate(examples)
        ]

        return self.apply(inputs, callbacks=callbacks)

    def _prepare_output(self, result: dict) -> dict:
        parsed_result = _parse_string_eval_output(result[self.output_key])
        if RUN_KEY in result:
            parsed_result[RUN_KEY] = result[RUN_KEY]
        return parsed_result

    def _evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        result = self(
            {
                "query": input,
                "context": reference,
                "result": prediction,
            },
            callbacks=callbacks,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)

    async def _aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        include_run_info: bool = False,
        **kwargs: Any,
    ) -> dict:
        result = await self.acall(
            inputs={"query": input, "context": reference, "result": prediction},
            callbacks=callbacks,
            include_run_info=include_run_info,
        )
        return self._prepare_output(result)


class CotQAEvalChain(ContextQAEvalChain):
    """LLM Chain for evaluating QA using chain of thought reasoning."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return "COT Contextual Accuracy"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> CotQAEvalChain:
        """Load QA Eval Chain from LLM."""
        prompt = prompt or COT_PROMPT
        cls._validate_input_vars(prompt)
        return cls(llm=llm, prompt=prompt, **kwargs)

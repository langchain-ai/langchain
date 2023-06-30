"""LLM Chain specifically for evaluating question answering."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from langchain import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.eval_prompt import CONTEXT_PROMPT, COT_PROMPT, PROMPT


def _parse_string_eval_output(text: str) -> dict:
    """Parse the output text.

    Args:
        text (str): The output text to parse.

    Returns:
        Any: The parsed output.
    """
    reasoning, verdict = text.strip().rsplit("\n", maxsplit=1)
    score = (
        1
        if verdict.upper() == "CORRECT"
        else (0 if verdict.upper() == "INCORRECT" else None)
    )
    return {
        "reasoning": reasoning.strip(),
        "value": verdict,
        "score": score,
    }


class QAEvalChain(LLMChain):
    """LLM Chain specifically for evaluating question answering."""

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, prompt: PromptTemplate = PROMPT, **kwargs: Any
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

    def evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate Chain or LLM output, based on optional input and label.

        Args:
            prediction (str): the LLM or chain prediction to evaluate.
            reference (Optional[str], optional): the reference label
                to evaluate against.
            input (Optional[str], optional): the input to consider during evaluation
            callbacks (Callbacks, optional): the callbacks to use for tracing.
            **kwargs: additional keyword arguments, including callbacks, tags, etc.
        Returns:
            dict: The evaluation results containing the score or value.
        """
        result = self.evaluate(
            examples=[{"query": input, "answer": reference}],
            predictions=[{"result": prediction}],
            callbacks=callbacks,
        )[0]
        return _parse_string_eval_output(result["text"])

    async def aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> dict:
        result = await self.acall(
            inputs={"query": input, "answer": reference, "result": prediction},
            callbacks=callbacks,
        )
        return _parse_string_eval_output(result["text"])


class ContextQAEvalChain(LLMChain):
    """LLM Chain specifically for evaluating QA w/o GT based on context"""

    @classmethod
    def _validate_input_vars(cls, prompt: PromptTemplate) -> None:
        expected_input_vars = {"query", "context", "result"}
        if expected_input_vars != set(prompt.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt.input_variables}"
            )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: PromptTemplate = CONTEXT_PROMPT,
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

    def evaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        result = self.evaluate(
            examples=[{"query": input, "context": reference}],
            predictions=[{"result": prediction}],
            callbacks=kwargs.get("callbacks"),
        )[0]
        return _parse_string_eval_output(result["text"])

    async def aevaluate_strings(
        self,
        *,
        prediction: str,
        reference: Optional[str] = None,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        result = await self.acall(
            inputs={"query": input, "context": reference, "result": prediction},
            callbacks=kwargs.get("callbacks"),
        )
        return _parse_string_eval_output(result["text"])


class CotQAEvalChain(ContextQAEvalChain):
    """LLM Chain specifically for evaluating QA using chain of thought reasoning."""

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, prompt: PromptTemplate = COT_PROMPT, **kwargs: Any
    ) -> CotQAEvalChain:
        cls._validate_input_vars(prompt)
        return cls(llm=llm, prompt=prompt, **kwargs)

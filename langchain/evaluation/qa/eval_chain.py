"""LLM Chain specifically for evaluating question answering."""
from __future__ import annotations

from typing import Any, List

from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.evaluation.qa.eval_prompt import PROMPT
from langchain.llms.base import BaseLLM


class QAEvalChain(LLMChain):
    """LLM Chain specifically for evaluating question answering."""

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, prompt: PromptTemplate = PROMPT, **kwargs: Any
    ) -> QAEvalChain:
        """Load QA Eval Chain from LLM.

        Args:
            llm (BaseLLM): the base language model to use.

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
        examples: List[dict],
        predictions: List[dict],
        question_key: str = "query",
        answer_key: str = "answer",
        prediction_key: str = "result",
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

        return self.apply(inputs)

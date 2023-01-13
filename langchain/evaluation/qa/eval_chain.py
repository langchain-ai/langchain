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
        """Load QA Eval Chain from LLM."""
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
        inputs = []
        if self.prompt != PROMPT:
            prompt_keys = self.prompt.input_variables
            input_keys = [question_key, answer_key, prediction_key]
            if set(prompt_keys) != set(input_keys):
                raise ValueError(
                    f"Input keys {input_keys} do not match prompt keys {prompt_keys}"
                )
            for i, example in enumerate(examples):
                _input = {
                    question_key: example[question_key],
                    answer_key: example[answer_key],
                    prediction_key: predictions[i][prediction_key],
                }
                inputs.append(_input)
        else:
            for i, example in enumerate(examples):
                _input = {
                    "query": example[question_key],
                    "answer": example[answer_key],
                    "result": predictions[i][prediction_key],
                }
                inputs.append(_input)

        return self.apply(inputs)

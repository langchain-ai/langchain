"""LLM Chain specifically for evaluating question answering."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.chat.base import BaseChatChain
from langchain.chat_models.base import BaseChatModel
from langchain.evaluation.qa.eval_prompt import (
    CHAT_INSTRUCTIONS,
    CHAT_RESPONSE_TEMPLATE,
)
from langchain.schema import ChatMessage


class ChatPromptTemplate(BaseModel, ABC):
    input_variables: List[str]

    @abstractmethod
    def format(self, *, query: str, answer: str, result: str) -> List[ChatMessage]:
        """Format chat prompt template."""


class EvalPrompt(ChatPromptTemplate):
    input_variables: List[str] = ["query", "answer", "result"]

    def format(
        self, *, query: str, answer: str, result: str, **kwargs: Any
    ) -> List[ChatMessage]:
        return [
            ChatMessage(text=CHAT_INSTRUCTIONS, role="system"),
            ChatMessage(
                text=CHAT_RESPONSE_TEMPLATE.format(
                    query=query, answer=answer, result=result
                ),
                role="user",
            ),
        ]


PROMPT = EvalPrompt()


class QAEvalChatChain(BaseChatChain):
    """Chat Chain specifically for evaluating question answering."""

    model: BaseChatModel
    prompt: ChatPromptTemplate

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return ["text"]

    @classmethod
    def from_model(
        cls, model: BaseChatModel, prompt: ChatPromptTemplate = PROMPT, **kwargs: Any
    ) -> QAEvalChatChain:
        expected_input_vars = {"query", "answer", "result"}
        if expected_input_vars != set(prompt.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt.input_variables}"
            )
        return cls(model=model, prompt=prompt, **kwargs)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        messages = self.prompt.format(**inputs)
        result = self.model.run(messages)
        return {"text": result.text}

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
        results = [self(inp) for inp in inputs]
        return results

from __future__ import annotations

from typing import Optional, Sequence, List

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.schema import PromptValue, LLMResult, BaseMessage


class ToyLLM(BaseLanguageModel):
    """A toy language model."""

    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None) -> str:
        """Predict the next token."""
        return text + "\n" + str(len(text))

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        pass

    def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        pass

    def predict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[Sequence[str]] = None
    ) -> BaseMessage:
        pass

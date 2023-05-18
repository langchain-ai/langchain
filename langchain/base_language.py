"""Base class for all language models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Set

from pydantic import BaseModel

from langchain.callbacks.manager import Callbacks
from langchain.schema import BaseMessage, LLMResult, PromptValue, get_buffer_string


def _get_num_tokens_default_method(text: str) -> int:
    """Get the number of tokens present in the text."""
    # TODO: this method may not be exact.
    # TODO: this method may differ based on model (eg codex).
    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        raise ValueError(
            "Could not import transformers python package. "
            "This is needed in order to calculate get_num_tokens. "
            "Please install it with `pip install transformers`."
        )
    # create a GPT-2 tokenizer instance
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # tokenize the text using the GPT-2 tokenizer
    tokenized_text = tokenizer.tokenize(text)

    # calculate the number of tokens in the tokenized text
    return len(tokenized_text)


class BaseLanguageModel(BaseModel, ABC):
    @abstractmethod
    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    @abstractmethod
    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    @abstractmethod
    def predict(self, text: str, *, stop: Optional[Sequence[str]] = None) -> str:
        """Predict text from text."""

    @abstractmethod
    def predict_messages(
        self, messages: List[BaseMessage], *, stop: Optional[Sequence[str]] = None
    ) -> BaseMessage:
        """Predict message from messages."""

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        return _get_num_tokens_default_method(text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in the message."""
        return sum([self.get_num_tokens(get_buffer_string([m])) for m in messages])

    @classmethod
    def all_required_field_names(cls) -> Set:
        all_required_field_names = set()
        for field in cls.__fields__.values():
            all_required_field_names.add(field.name)
            if field.has_alias:
                all_required_field_names.add(field.alias)
        return all_required_field_names

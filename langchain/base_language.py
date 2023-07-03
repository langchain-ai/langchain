"""Base class for all language models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Set

from langchain.callbacks.manager import Callbacks
from langchain.load.serializable import Serializable
from langchain.schema import BaseMessage, LLMResult, PromptValue, get_buffer_string


class BaseLanguageModel(Serializable, ABC):
    """Abstract base class for interfacing with all language models.

    All language model wrappers inherit from BaseLanguageModel.

    Exposes three main methods:
    - generate_prompt: generate language model outputs for a sequence of prompt
        values. A prompt value is a model input that can be converted to any language
        model input format (string or messages).
    - predict: pass in a single string to a language model and return a string
        prediction.
    - predict_messages: pass in a sequence of BaseMessages (corresponding to a single
        model call) to a language model and return a BaseMessage prediction.

    Each of these has an equivalent asynchronous method.
    """

    @abstractmethod
    def generate_prompt(
        self,
        prompts: Sequence[PromptValue],
        stop: Optional[Sequence[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Take in a sequence of PromptValue and return an LLMResult.

        This method should make use of batched calls for model that expose a batched
        API.

        Args:
            prompts: Sequence of prompt values. A PromptValue is an object that can be
                converted to match the format of any language model (string for pure
                text generation models and BaseMessages for chat models).
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callback handlers for ex
            kwargs: Arbitrary additional keyword arguments.
        """

    @abstractmethod
    async def agenerate_prompt(
        self,
        prompts: Sequence[PromptValue],
        stop: Optional[Sequence[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    @abstractmethod
    def predict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        """Predict text from text."""

    @abstractmethod
    def predict_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Predict message from messages."""

    @abstractmethod
    async def apredict(
        self, text: str, *, stop: Optional[Sequence[str]] = None, **kwargs: Any
    ) -> str:
        """Predict text from text."""

    @abstractmethod
    async def apredict_messages(
        self,
        messages: Sequence[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Predict message from messages."""

    def get_token_ids(self, text: str) -> List[int]:
        """Get the token present in the text."""
        return _get_token_ids_default_method(text)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        return len(self.get_token_ids(text))

    def get_num_tokens_from_messages(self, messages: Sequence[BaseMessage]) -> int:
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


def _get_token_ids_default_method(text: str) -> List[int]:
    """Encode the text into token IDs."""
    # TODO: this method may not be exact.
    # TODO: this method may differ based on model (eg codex).
    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        raise ValueError(
            "Could not import transformers python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install transformers`."
        )
    # create a GPT-2 tokenizer instance
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # tokenize the text using the GPT-2 tokenizer
    return tokenizer.encode(text)

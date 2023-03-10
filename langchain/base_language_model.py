"""Base class for language models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Extra, Field, validator

import langchain
from langchain.callbacks import BaseCallbackManager, get_callback_manager
from langchain.schema import LLMResult, PromptValue


def _get_verbosity() -> bool:
    return langchain.verbose


class BaseLanguageModel(BaseModel, ABC):
    """Base class for language models."""

    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """If callback manager is None, set it.

        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

    @validator("verbose", pre=True, always=True)
    def set_verbose(cls, verbose: Optional[bool]) -> bool:
        """If verbose is None, set it.

        This allows users to pass in None as verbose to access the global setting.
        """
        if verbose is None:
            return _get_verbosity()
        else:
            return verbose

    def generate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""
        self.callback_manager.on_llm_start_prompt_value(
            {"name": self.__class__.__name__}, prompts, verbose=self.verbose
        )
        try:
            output = self._generate_prompt(prompts, stop=stop)
        except (KeyboardInterrupt, Exception) as e:
            self.callback_manager.on_llm_error(e, verbose=self.verbose)
            raise e
        self.callback_manager.on_llm_end(output, verbose=self.verbose)
        return output

    async def agenerate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""
        if self.callback_manager.is_async:
            await self.callback_manager.on_llm_start_prompt_value(
                {"name": self.__class__.__name__}, prompts, verbose=self.verbose
            )
        else:
            self.callback_manager.on_llm_start_prompt_value(
                {"name": self.__class__.__name__}, prompts, verbose=self.verbose
            )
        try:
            output = await self._agenerate_prompt(prompts, stop=stop)
        except (KeyboardInterrupt, Exception) as e:
            if self.callback_manager.is_async:
                await self.callback_manager.on_llm_error(e, verbose=self.verbose)
            else:
                self.callback_manager.on_llm_error(e, verbose=self.verbose)
            raise e
        if self.callback_manager.is_async:
            await self.callback_manager.on_llm_end(output, verbose=self.verbose)
        else:
            self.callback_manager.on_llm_end(output, verbose=self.verbose)
        return output

    @abstractmethod
    def _generate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    @abstractmethod
    async def _agenerate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        # TODO: this method may not be exact.
        # TODO: this method may differ based on model (eg codex).
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install transformers`."
            )
        # create a GPT-3 tokenizer instance
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # tokenize the text using the GPT-3 tokenizer
        tokenized_text = tokenizer.tokenize(text)

        # calculate the number of tokens in the tokenized text
        return len(tokenized_text)

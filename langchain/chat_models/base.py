import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Extra, Field, validator

import langchain
from langchain.callbacks import get_callback_manager
from langchain.callbacks.base import BaseCallbackManager
from langchain.schema import (
    AIMessage,
    BaseLanguageModel,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    LLMResult,
    PromptValue,
)


def _get_verbosity() -> bool:
    return langchain.verbose


class BaseChatModel(BaseLanguageModel, BaseModel, ABC):
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

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return {}

    def generate(
        self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Top Level call"""
        results = [self._generate(m, stop=stop) for m in messages]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        return LLMResult(generations=generations, llm_output=llm_output)

    async def agenerate(
        self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Top Level call"""
        results = await asyncio.gather(
            *[self._agenerate(m, stop=stop) for m in messages]
        )
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        return LLMResult(generations=generations, llm_output=llm_output)

    def generate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        prompt_strings = [p.to_string() for p in prompts]
        self.callback_manager.on_llm_start(
            {"name": self.__class__.__name__}, prompt_strings, verbose=self.verbose
        )
        try:
            output = self.generate(prompt_messages, stop=stop)
        except (KeyboardInterrupt, Exception) as e:
            self.callback_manager.on_llm_error(e, verbose=self.verbose)
            raise e
        self.callback_manager.on_llm_end(output, verbose=self.verbose)
        return output

    async def agenerate_prompt(
        self, prompts: List[PromptValue], stop: Optional[List[str]] = None
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        prompt_strings = [p.to_string() for p in prompts]
        if self.callback_manager.is_async:
            await self.callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, prompt_strings, verbose=self.verbose
            )
        else:
            self.callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, prompt_strings, verbose=self.verbose
            )
        try:
            output = await self.agenerate(prompt_messages, stop=stop)
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
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Top Level call"""

    @abstractmethod
    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        """Top Level call"""

    def __call__(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> BaseMessage:
        return self._generate(messages, stop=stop).generations[0].message

    def call_as_llm(self, message: str, stop: Optional[List[str]] = None) -> str:
        result = self([HumanMessage(content=message)], stop=stop)
        return result.content


class SimpleChatModel(BaseChatModel):
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> str:
        """Simpler interface."""

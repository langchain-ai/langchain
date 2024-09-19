import asyncio
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableConfig


class FakeListLLM(LLM):
    """Fake LLM for testing purposes."""

    responses: list[str]
    """List of responses to return in order."""
    # This parameter should be removed from FakeListLLM since
    # it's only used by sub-classes.
    sleep: Optional[float] = None
    """Sleep time in seconds between responses.
    
    Ignored by FakeListLLM, but used by sub-classes.
    """
    i: int = 0
    """Internally incremented after every model invocation.
    
    Useful primarily for testing purposes.
    """

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-list"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": self.responses}


class FakeListLLMError(Exception):
    """Fake error for testing purposes."""


class FakeStreamingListLLM(FakeListLLM):
    """Fake streaming list LLM for testing purposes.

    An LLM that will return responses from a list in order.

    This model also supports optionally sleeping between successive
    chunks in a streaming implementation.
    """

    error_on_chunk_number: Optional[int] = None
    """If set, will raise an exception on the specified chunk number."""

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        result = self.invoke(input, config)
        for i_c, c in enumerate(result):
            if self.sleep is not None:
                time.sleep(self.sleep)

            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListLLMError
            yield c

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        result = await self.ainvoke(input, config)
        for i_c, c in enumerate(result):
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)

            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListLLMError
            yield c

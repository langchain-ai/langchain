"""Fake LLMs for testing purposes."""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from typing_extensions import override

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
    sleep: float | None = None
    """Sleep time in seconds between responses.

    Ignored by FakeListLLM, but used by sub-classes.
    """
    i: int = 0
    """Internally incremented after every model invocation.

    Useful primarily for testing purposes.
    """

    @property
    @override
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-list"

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Return next response."""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @override
    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Return next response."""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @property
    @override
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

    error_on_chunk_number: int | None = None
    """If set, will raise an exception on the specified chunk number."""

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
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

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
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

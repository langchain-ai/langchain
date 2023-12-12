import asyncio
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult


class __ModuleName__LLM(BaseLLM):
    """__ModuleName__LLM large language models.

    Example:
        .. code-block:: python

            from __module_name__ import __ModuleName__LLM

            model = __ModuleName__LLM()
    """

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "__package_name_short__-llm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        raise NotImplementedError

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        # Change implementation if integration natively supports async generation.
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._generate, **kwargs), prompts, stop, run_manager
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        raise NotImplementedError

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(text="Yield chunks")
        yield GenerationChunk(text=" like this!")

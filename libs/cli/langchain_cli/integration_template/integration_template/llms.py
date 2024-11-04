"""__ModuleName__ large language models."""

from typing import (
    Any,
    List,
    Optional,
)

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult


class __ModuleName__LLM(BaseLLM):
    """__ModuleName__ completion model integration.

    # TODO: Replace with relevant packages, env vars.
    Setup:
        Install ``__package_name__`` and set environment variable ``__MODULE_NAME___API_KEY``.

        .. code-block:: bash

            pip install -U __package_name__
            export __MODULE_NAME___API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args — completion params:
        model: str
            Name of __ModuleName__ model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    # TODO: Populate with relevant params.
    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            __ModuleName__ API key. If not passed in will be read from env var __MODULE_NAME___API_KEY.

    See full list of supported init args and their descriptions in the params section.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from __module_name__ import __ModuleName__LLM

            llm = __ModuleName__LLM(
                model="...",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            input_text = "The meaning of life is "
            llm.invoke(input_text)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if token-level streaming isn't supported.
    Stream:
        .. code-block:: python

            for chunk in llm.stream(input_text):
                print(chunk)

        .. code-block:: python

            # TODO: Example output.

        .. code-block:: python

            ''.join(llm.stream(input_text))

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if native async isn't supported.
    Async:
        .. code-block:: python

            await llm.ainvoke(input_text)

            # stream:
            # async for chunk in (await llm.astream(input_text))

            # batch:
            # await llm.abatch([input_text])

        .. code-block:: python

            # TODO: Example output.
    """

    # TODO: This method must be implemented to generate text completions.
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        raise NotImplementedError

    # TODO: Implement if __ModuleName__LLM supports async generation. Otherwise
    # delete method.
    # async def _agenerate(
    #     self,
    #     prompts: List[str],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> LLMResult:
    #     raise NotImplementedError

    # TODO: Implement if __ModuleName__LLM supports streaming. Otherwise delete method.
    # def _stream(
    #     self,
    #     prompt: str,
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> Iterator[GenerationChunk]:
    #     raise NotImplementedError

    # TODO: Implement if __ModuleName__LLM supports async streaming. Otherwise delete
    # method.
    # async def _astream(
    #     self,
    #     prompt: str,
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[GenerationChunk]:
    #     raise NotImplementedError

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "__package_name_short__-llm"

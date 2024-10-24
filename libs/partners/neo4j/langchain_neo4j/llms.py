"""Neo4j large language models."""

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


class Neo4jLLM(BaseLLM):
    """Neo4j completion model integration.

    # TODO: Replace with relevant packages, env vars.
    Setup:
        Install ``langchain-neo4j`` and set environment variable ``NEO4J_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-neo4j
            export NEO4J_API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args — completion params:
        model: str
            Name of Neo4j model to use.
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
            Neo4j API key. If not passed in will be read from env var NEO4J_API_KEY.

    See full list of supported init args and their descriptions in the params section.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_neo4j import Neo4jLLM

            llm = Neo4jLLM(
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

    # TODO: Implement if Neo4jLLM supports async generation. Otherwise
    # delete method.
    # async def _agenerate(
    #     self,
    #     prompts: List[str],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> LLMResult:
    #     raise NotImplementedError

    # TODO: Implement if Neo4jLLM supports streaming. Otherwise delete method.
    # def _stream(
    #     self,
    #     prompt: str,
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> Iterator[GenerationChunk]:
    #     raise NotImplementedError

    # TODO: Implement if Neo4jLLM supports async streaming. Otherwise delete
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
        return "neo4j-llm"
